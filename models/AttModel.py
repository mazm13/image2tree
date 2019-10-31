from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph.node import TensorNode, Node
from graph.ternary import Tree


class ConditionalCNN_v2(nn.Module):
    def __init__(self, opt):
        super(ConditionalCNN_v2, self).__init__()
        self.rnn_size = opt.hidden_size
        self.att_hid_size = opt.att_hid_size

        self.conv1 = nn.Conv2d(self.att_hid_size, self.att_hid_size // 16, 3, stride=1, padding=1)  # 14*14*64
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.att_hid_size // 16, 1, 3, stride=1, padding=1)  # 14*14*1
        # self.soft = nn.Softmax2d()

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def forward(self, h, att_feats):
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)

        x = att + att_h
        x = x.view(-1, 14, 14, self.att_hid_size)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 2, 3)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        weight = x.view(-1, att_size)
        weight = F.softmax(weight)

        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)

        return att_res


class Attention(nn.Module):
    # Spatial Attention Model 
    # from "Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning"
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.opt = opt
        self.W_v = nn.Linear(opt.embed_size, opt.att_size)
        self.W_g = nn.Linear(opt.hidden_size, opt.att_size)
        self.W_h = nn.Linear(opt.att_size, 1)

    def forward(self, h, att_feats):
        # input, h_t, V
        # z_t = w_h^T tanh(W_v V + (W_g h_t)1^T)
        # \alpha_t = softmax(z_t)
        # c_t = \sum_{i=1}^k \alpha_{ti} v_{ti}
        # return c_t

        # att_feats: batch_size * num_feats * embed_size
        # h: batch_size * hidden_size

        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att_v = self.W_v(att_feats)  # batch_size * att_size * att_size
        h_g = self.W_g(h)  # batch_size * att_size
        h_g = h_g.unsqueeze(1).expand(h_g.size(0), att_size, h_g.size(1))

        zeta = self.W_h(F.tanh(att_v + h_g))  # add with breadcast, batch_size * att_size * 1
        zeta = torch.squeeze(zeta, dim=2)
        alpha = F.softmax(zeta, dim=1)

        cont = torch.bmm(alpha.unsqueeze(1), att_feats)
        return cont.squeeze(dim=1)


class AttModel(nn.Module):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.opt = opt
        self.gpu = opt.use_cuda

        self.fc_embed = nn.Linear(opt.fc_feat_size, opt.embed_size)
        self.att_embed = nn.Linear(opt.att_feat_size, opt.embed_size)
        self.attention = Attention(opt)
        # self.attention = ConditionalCNN_v2(opt)
        self.embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        self.cell_a = nn.GRUCell(opt.embed_size, opt.hidden_size)
        self.cell_f = nn.GRUCell(opt.embed_size, opt.hidden_size)
        self.linear_a = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_f = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.logit = nn.Linear(opt.hidden_size + opt.embed_size, opt.vocab_size)

    def init_hidden(self, bsz):
        h = torch.zeros(bsz, self.opt.hidden_size, requires_grad=False)
        if self.gpu:
            h = h.cuda()
        return h

    def set_gpu(self, gpu):
        self.gpu = gpu

    def _forward(self, node):

        if node is None:
            return
        h_a = node.parent.h
        h_a = self.cell_a(node.parent.word_embed, h_a)

        if node.left_brother is not None:
            h_f = node.left_brother.h
            h_f = self.cell_f(node.left_brother.word_embed, h_f)
        else:
            h_f = self.init_hidden(h_a.size(0))

        h = F.tanh(self.linear_a(h_a) + self.linear_f(h_f))
        cont = self.attention(h, self.att_feats)
        node.h = h
        node.cont = cont
        node.word_embed = self.embed(node.word_idx)

        for c in [node.lc, node.mc, node.rc]:
            self._forward(c)

    def forward(self, tree, fc_feat, att_feats):
        bsz = fc_feat.size(0)

        h_a = self.init_hidden(bsz)
        fc_embed = self.fc_embed(fc_feat)
        h_a = self.cell_a(fc_embed, h_a)
        h_f = self.init_hidden(bsz)
        h = F.tanh(self.linear_a(h_a) + self.linear_f(h_f))

        att_feats = self.att_embed(att_feats)
        # print("att_feats with size {}".format(att_feats))
        self.att_feats = att_feats

        # print(h.size())
        # print(att_feats.size())

        cont = self.attention(h, att_feats)
        node = tree.root
        node.h = h
        node.cont = cont
        node.word_embed = self.embed(node.word_idx)

        for c in [node.lc, node.mc, node.rc]:
            self._forward(c)

        target = [n.word_idx for n in tree.nodes.values()]
        target = torch.stack(target, dim=1)

        hs = [n.h for n in tree.nodes.values()]
        hs = torch.stack(hs, dim=1)
        conts = [n.cont for n in tree.nodes.values()]
        conts = torch.stack(conts, dim=1)
        output = self.logit(torch.cat([hs, conts], dim=2))

        target = target.view(-1)
        output = output.view(-1, output.size(-1))
        return output, target

    def from_scratch(self, fc_feat, att_feats):
        h_a = self.init_hidden(1)
        fc_embed = self.fc_embed(fc_feat)
        h_a = self.cell_a(fc_embed, h_a)
        h_f = self.init_hidden(1)
        h = F.tanh(self.linear_a(h_a) + self.linear_f(h_f))

        att_feats = self.att_embed(att_feats)
        self.att_feats = att_feats
        cont = self.attention(h, att_feats)

        logits = self.logit(torch.cat([h, cont], dim=1))
        probs = F.softmax(logits, dim=1)
        return h, probs

    def step(self, node):
        h_a = node.parent.h
        word_embed_a = self.embed(node.parent.word_idx)
        h_a = self.cell_a(word_embed_a, h_a)

        if node.left_brother is not None:
            h_f = node.left_brother.h
            word_embed_f = self.embed(node.left_brother.word_idx)
            h_f = self.cell_f(word_embed_f, h_f)
        else:
            h_f = self.init_hidden(1)

        h = F.tanh(self.linear_a(h_a) + self.linear_f(h_f))
        cont = self.attention(h, self.att_feats)

        logits = self.logit(torch.cat([h, cont], dim=1))
        probs = F.softmax(logits, dim=1)
        return h, probs

    def greedy_search(self, fc_feat, att_feats, vocab, max_seq_length):
        h_r, probs_r = self.from_scratch(fc_feat, att_feats)
        _, word_idx_r = probs_r.max(dim=1)
        lex_r = vocab.idx2word[word_idx_r.item()]
        word_idx_r = torch.LongTensor([word_idx_r])

        node_r = TensorNode(Node(lex_r, 1, ""))
        node_r.word_idx = word_idx_r
        node_r.h = h_r

        tree = Tree()
        tree.root = node_r
        tree.nodes.update({node_r.idx - 1: node_r})

        tree.model = self
        tree.vocab = vocab
        tree.max_seq_length = max_seq_length
        tree.bfs_breed()
        return tree
