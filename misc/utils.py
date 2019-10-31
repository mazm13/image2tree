import torch
import torch.nn as nn
from queue import Queue


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def to_contiguous(tensor):
    # return tensor if tensor.is_contiguous() else tensor.contiguous()
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class Node(object):

    def __init__(self, value):
        self.value = value
        self.childern = []


def tree2seq(node, sent):
    if node.value == 1:
        return
    if len(node.childern) == 0:
        sent.append(node.value)
    else:
        tree2seq(node.childern[0], sent)
        sent.append(node.value)
        tree2seq(node.childern[1], sent)
        tree2seq(node.childern[2], sent)


def decode_sequence(vocab, word_idx, father_idx, mask):
    bsz = word_idx.size(0)
    max_len = word_idx.size(1)
    sents = []
    for b in range(bsz):
        nodes = [Node(word_idx[b, 0].item())]
        for i in range(1, max_len):
            if mask[b][i] == 0:
                break
            node = Node(word_idx[b, i].item())
            nodes.append(node)
            nodes[father_idx[b][i].item()].childern.append(node)
        sent = []
        tree2seq(nodes[0], sent)
        sent = [vocab.idx2word[w] for w in sent]
        sents.append(' '.join(sent))
    return sents


def seq2mask(word_idx):
    masks = torch.zeros_like(word_idx).long()
    for i in range(word_idx.size(0)):
        c = 1
        stop = 0
        for j in range(word_idx.size(1)):
            if word_idx[i][j] == 1:
                c -= 1
            else:
                c += 2
            if j % 3 == 0 and c == 0:
                stop = j
                break
        masks[i, :stop+1] = 1
    return masks.cuda()


def seq2ratio(word_idx, father_idx, mask):
    ratio = torch.zeros_like(word_idx).float()
    for i in range(father_idx.size(0)):
        for j in range(1, torch.sum(mask[i])):
            ratio[i][j] = ratio[i][father_idx[i][j]] + 1.0
    return ratio


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, mask, reward, ratio):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = mask.float()
        # mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        mask = to_contiguous(mask).view(-1)
        ratio = to_contiguous(ratio).view(-1)
        ratio = torch.pow(0.95, ratio)
        # output = - input * reward * mask * ratio
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, logprobs, target, mask):
        # truncate to the same size
        batch_size = logprobs.size(0)
        target = target[:, :logprobs.size(1)]
        mask = mask[:, :logprobs.size(1)]
        # print(input.size())
        # print(target.size())
        # print(mask.size())

        # import math
        # loss = 0
        # tot = 0
        # for i in range(2):
        #     for j in range(40):
        #         _t = target[i][j].item()
        #         _logprob = input[i][j][_t].item()
        #         _mask = mask[i][j].item()
        #         print(math.exp(_logprob), _mask)
        #         loss += _logprob * _mask
        #         tot += _mask
        # print(loss / tot)

        # output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        # output = torch.sum(output) / torch.sum(mask)
        logprobs = to_contiguous(logprobs).view(-1, logprobs.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = -logprobs.gather(1, target) * mask
        # output = torch.sum(output) / batch_size
        output = torch.sum(output) / torch.sum(mask)

        return output
