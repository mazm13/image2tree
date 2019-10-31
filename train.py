from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import json
from dataset import SeqtreeCOCO

from misc import utils
from misc.logger import Logger
from misc.utils import LanguageModelCriterion
import eval_utils
import eval_seqtree
from misc.reward import get_self_critical_reward, init_scorer
from misc.utils import RewardCriterion


def train(model, vocab, cfg):
    seqtree_coco = SeqtreeCOCO()
    loader = DataLoader(seqtree_coco, batch_size=16, shuffle=True, num_workers=4)
    logdir = os.path.join(cfg.checkpoint_path, cfg.id)
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    logger = Logger(logdir)

    with open(os.path.join(logdir, 'config.txt'), 'w') as f:
        f.write(str(cfg))
    with open('data/idx2caps.json', 'r') as f:
        cocoid2caps = json.load(f)
    cocoid2caps = {int(k): v for k, v in cocoid2caps.items()}
    init_scorer('coco-train-idxs')

    infos = {}
    # if cfg.start_from is not None:
    #     with open(os.path.join(cfg.start_from, 'infos_' + cfg.start_from + '_best.pkl'), 'rb') as f:
    #         infos = pickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})

    best_val_score = 0

    update_lr_flag = True
    if cfg.caption_model == 'att_model' or cfg.caption_model == 'tree_model' \
            or cfg.caption_model == 'tree_model_1' or cfg.caption_model == 'tree_model_md' \
            or cfg.caption_model == 'tree_model_2' or cfg.caption_model == 'tree_model_md_att' \
            or cfg.caption_model == 'tree_model_md_sob' or cfg.caption_model == 'tree_model_md_in' \
            or cfg.caption_model == 'drnn':
        # crit = nn.CrossEntropyLoss()
        crit = LanguageModelCriterion()
        rl_crit = RewardCriterion()
    else:
        raise Exception("Caption model not supported: {}".format(cfg.caption_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    num_period_best = 0
    current_score = 0
    start = time.time()

    print("start training...")

    while True:
        if update_lr_flag:
            if epoch > cfg.learning_rate_decay_start >= 0:
                frac = (epoch - cfg.learning_rate_decay_start) // cfg.learning_rate_decay_every
                decay_factor = cfg.learning_rate_decay_rate ** frac
                cfg.current_lr = cfg.learning_rate * decay_factor
                utils.set_lr(optimizer, cfg.current_lr)
            else:
                cfg.current_lr = cfg.learning_rate

        optimizer.zero_grad()
        for data in loader:
            if cfg.use_cuda:
                torch.cuda.synchronize()

            if cfg.caption_model == 'tree_model_md_att':
                temp = [data['word_idx'], data['father_idx'], data['masks'], data['fc_feats'], data['att_feats']]
                temp = [_.cuda() for _ in temp]
                word_idx, father_idx, masks, fc_feats, att_feats = temp

            elif cfg.caption_model == 'tree_model_md' or cfg.caption_model == 'tree_model_md_sob' \
                    or cfg.caption_model == 'tree_model_md_in' or cfg.caption_model == 'drnn':
                temp = [data['word_idx'], data['father_idx'], data['masks'], data['fc_feats']]
                temp = [_.cuda() for _ in temp]
                word_idx, father_idx, masks, fc_feats = temp
                # words = [[vocab.idx2word[word_idx[batch_index][i].item()] for i in range(40)]
                #          for batch_index in range(2)]

            else:
                raise Exception("Caption model not supported: {}".format(cfg.caption_model))

            optimizer.zero_grad()
            # if cfg.caption_model == 'tree_model_md_att':
            #     logprobs = model(word_idx, father_idx, fc_feats, att_feats)
            #     loss = crit(logprobs, word_idx, masks)
            if cfg.caption_model == 'tree_model_md' or cfg.caption_model == 'tree_model_md_sob' \
                    or cfg.caption_model == 'tree_model_md_in' or cfg.caption_model == 'drnn' \
                    or cfg.caption_model == 'tree_model_md_att':
                word_idx, father_idx, mask, seqLogprobs = model._sample(fc_feats, att_feats, max_seq_length=40)
                gen_result = utils.decode_sequence(vocab, word_idx, father_idx, mask)
                ratio = utils.seq2ratio(word_idx, father_idx, mask)
                reward = get_self_critical_reward(model, fc_feats, att_feats, data, gen_result, vocab, cocoid2caps,
                                                  word_idx.size(1), cfg)
                loss = rl_crit(seqLogprobs, mask, torch.from_numpy(reward).float().cuda(), ratio)

            else:
                raise Exception("Caption model not supported: {}".format(cfg.caption_model))

            loss.backward()
            utils.clip_gradient(optimizer, cfg.grad_clip)
            optimizer.step()
            train_loss = loss.item()

            if cfg.use_cuda:
                torch.cuda.synchronize()

            if iteration % cfg.losses_log_every == 0:
                end = time.time()
                logger.scalar_summary('train_loss', train_loss, iteration)
                logger.scalar_summary('learning_rate', cfg.learning_rate, iteration)
                loss_history[iteration] = train_loss
                lr_history[iteration] = cfg.current_lr
                print(
                    "iter {} (epoch {}), learning_rate: {:.6f}, train_loss: {:.6f}, current_cider: {:.3f}, best_cider: {:.3f}, time/log = {:.3f}" \
                        .format(iteration, epoch, cfg.current_lr, train_loss, current_score, best_val_score,
                                end - start))
                start = time.time()

            if (iteration + 1) % cfg.save_checkpoint_every == 0:
                eval_kwargs = {'eval_split': 'val', 'eval_time': False}
                eval_kwargs.update(vars(cfg))
                # lang_stats = eval_utils.eval_split(model, vocab, eval_kwargs)
                lang_stats = eval_seqtree.eval_split(model, vocab, eval_kwargs)
                if cfg.use_cuda:
                    model = model.cuda()

                for k, v in lang_stats.items():
                    logger.scalar_summary(k, v, iteration)

                val_result_history[iteration] = {'lang_stats': lang_stats}

                current_score = lang_stats['CIDEr']
                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                    num_period_best = 1
                else:
                    num_period_best += 1

                if best_flag:
                    infos['iter'] = iteration
                    infos['epoch'] = epoch
                    infos['val_result_history'] = val_result_history
                    infos['loss_history'] = loss_history
                    infos['lr_history'] = lr_history

                    checkpoint_path = os.path.join(logdir, 'model_' + cfg.id + '_best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    optimizer_path = os.path.join(logdir, 'optimizer_' + cfg.id + '_best.pth')
                    torch.save(optimizer.state_dict(), optimizer_path)
                    print("model saved to {}".format(logdir))
                    with open(os.path.join(logdir, 'infos_' + cfg.id + '_best.pkl'), 'wb') as f:
                        pickle.dump(infos, f)

                if num_period_best >= cfg.num_eval_no_improve:
                    print('no improvement, exit({})'.format(best_val_score))
                    sys.exit()

            iteration += 1

        epoch += 1
        if epoch >= cfg.max_epoches != -1:
            break
