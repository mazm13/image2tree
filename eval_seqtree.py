from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
from tqdm import tqdm
from dataloader import get_eval_loader
from misc import utils


def eval_split(model, vocab, eval_kwargs):
    loader = get_eval_loader(kwargs=eval_kwargs)
    print("assigned {} images for model evaluation in Karpathy {} split".format(
        len(loader), eval_kwargs['eval_split']))

    predictions = []
    eval_mode = eval_kwargs.get('eval_mode', 0)

    start = time.time()
    for i, batch in enumerate(loader):
        temp = [_.cuda() for _ in batch]
        cocoid, fc_feat, att_feat = temp
        word_idx, father_idx, mask = model._greedy_search(fc_feat, att_feat, 40)
        sents = utils.decode_sequence(vocab, word_idx, father_idx, mask)

        for j in range(len(sents)):
            entry = {'image_id': cocoid[j].item(), 'caption': sents[j]}
            print('{}: {}({})'.format(cocoid[j].item(), sents[j], i))
            predictions.append(entry)
        if i > eval_kwargs['eval_images'] >= 0:
            break
    print("inference took {} seconds.".format(time.time() - start))
    lang_stat = language_eval(predictions, eval_kwargs['id'], eval_kwargs['eval_split'])

    if not eval_kwargs['eval_time']:
        model.train()
        model.set_gpu(eval_kwargs['use_cuda'] == 1)
    return lang_stat


def language_eval(preds, model_id, split):
    import sys
    sys.path.append('coco-caption')
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    coco = COCO(annFile)
    valids = coco.getImgIds()

    preds_filt = [p for p in preds if p['image_id'] in valids]

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results', model_id + '_' + split + '_k3.json')
    with open(cache_path, 'w') as f:
        json.dump(preds_filt, f)

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    eval_results_path = os.path.join('eval_results', model_id + '_' + split + '_lng.json')
    with open(eval_results_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    return out
