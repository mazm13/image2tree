from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
from tqdm import tqdm
from dataloader import get_eval_loader


def eval_split(model, vocab, eval_kwargs):
    loader = get_eval_loader(kwargs=eval_kwargs)
    print("assigned {} images for model evaluation in Karpathy {} split".format(
        len(loader), eval_kwargs['eval_split']))

    if not eval_kwargs['eval_time']:
        model = model.cpu()
        model.set_gpu(False)

    predictions = []

    eval_mode = eval_kwargs.get('eval_mode', 0)
    save_tree_shape = eval_kwargs.get('save_tree_shape', 0)

    if save_tree_shape > 0:
        cache_dir = os.path.join('eval_results', eval_kwargs['id'] + '_' + eval_kwargs['eval_split'])
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)

    start = time.time()
    for i, batch in enumerate(loader):
        if eval_kwargs['caption_model'] == 'att_model' or eval_kwargs['caption_model'] == 'tree_model_md_att':
            cocoid, fc_feat, att_feat = batch
            if eval_mode:
                ps, cs = model.beam_search(fc_feat, att_feat, vocab, eval_kwargs['max_seq_length'], 3, 3)
                if len(cs) == 0:
                    ps = sorted(ps, key=lambda x: -x.logprob)
                    tree = ps[0]
                else:
                    for _cs in cs:
                        _cs.logprob /= len(_cs.nodes) ** 0.9
                    cs = sorted(cs, key=lambda x: -x.logprob)
                    tree = cs[0]
            else:
                # tree = model.greedy_search(fc_feat, vocab, eval_kwargs['max_seq_length'])
                tree = model.greedy_search(fc_feat, att_feat, vocab, eval_kwargs['max_seq_length'])
        elif eval_kwargs['caption_model'] == 'tree_model' or eval_kwargs['caption_model'] == 'tree_model_1' \
                or eval_kwargs['caption_model'] == 'tree_model_md' or eval_kwargs['caption_model'] == 'tree_model_2'\
                or eval_kwargs['caption_model'] == 'tree_model_md_sob' \
                or eval_kwargs['caption_model'] == 'tree_model_md_in' \
                or eval_kwargs['caption_model'] == 'drnn':
            cocoid, fc_feat = batch
            if eval_mode:
                ps, cs = model.beam_search(fc_feat, vocab, eval_kwargs['max_seq_length'], 4, 4)
                if len(cs) == 0:
                    ps = sorted(ps, key=lambda x: -x.logprob)
                    tree = ps[0]
                else:
                    for _cs in cs:
                        _cs.logprob /= len(_cs.nodes) ** 0.9
                    cs = sorted(cs, key=lambda x: -x.logprob)
                    tree = cs[0]
            else:
                # tree = model.greedy_search(fc_feat, vocab, eval_kwargs['max_seq_length'])
                tree = model._sample(fc_feat, vocab, eval_kwargs['max_seq_length'])
        else:
            raise NotImplementedError
        if save_tree_shape > 0:
            with open(os.path.join(cache_dir, str(cocoid)), 'w') as f:
                f.write(tree.graphviz())
        caption = tree.__str__()
        entry = {'image_id': cocoid, 'caption': caption}
        print("({}) {} ({}/5000)".format(cocoid, caption, i+1))
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
