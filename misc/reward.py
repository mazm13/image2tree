import torch
import utils
from collections import OrderedDict
import numpy as np

import sys
sys.path.append("sider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None


def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)


def get_self_critical_reward(model, fc_feats, att_feats, data, gen_result, vocab, cocoid2caps, seqLen, opt):
    # batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    batch_size = len(gen_result)
    # seq_per_img = batch_size // len(data['gts'])

    # get greedy decoding baseline
    model.eval()
    with torch.no_grad():
        # greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
        word_idx, father_idx, mask = model._greedy_search(fc_feats, att_feats, max_seq_length=40)
    model.train()
    greedy_res = utils.decode_sequence(vocab, word_idx, father_idx, mask)

    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [gen_result[i]]
    for i in range(batch_size):
        res[batch_size + i] = [greedy_res[i]]

    gts = OrderedDict()
    for i in range(batch_size):
        gts[i] = cocoid2caps[data['image_id'][i].item()]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}

    # for i in range(2 * batch_size):
    #     print(res[i], gts[i])

    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Sider scores:', _ * 0.1)
    else:
        cider_scores = 0

    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0

    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores
    scores = scores[:batch_size] - scores[batch_size:]
    scores = scores * 0.1
    print('Mean reward:', scores.mean())
    rewards = np.repeat(scores[:, np.newaxis], seqLen, 1)

    return rewards
