from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph.node import TensorNode, Node
from graph.ternary import Tree


class TreeModel1(nn.Module):
    def __init__(self, opt):
        super(TreeModel1, self).__init__()
        self.opt = opt

        self.fc_embed = nn.Linear(opt.fc_feat_size, opt.embed_size)

        self.embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        # ancestral (parent-to-children)
        #self.cell_a = nn.GRUCell(opt.embed_size, opt.hidden_size)
        self.cell_a = nn.LSTMCell(opt.embed_size, opt.hidden_size)
        # fraternal (sibling-to-sibling)
        # self.cell_f = nn.GRUCell(opt.embed_size, opt.hidden_size)
        self.cell_f = nn.LSTMCell(opt.embed_size, opt.hidden_size)

        self.linear_a = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_f = nn.Linear(opt.hidden_size, opt.hidden_size)

        self.logit = nn.Linear(opt.hidden_size, opt.vocab_size)
        self.gpu = opt.use_cuda

    def init_state(self, bsz):
        h = torch.zeros(bsz, self.opt.hidden_size, requires_grad=False)
        c = torch.zeros(bsz, self.opt.hidden_size, requires_grad=False)
        if self.gpu:
            h, c = h.cuda(), c.cuda()
        return h, c

    def set_gpu(self, gpu):
        self.gpu = gpu

    def _forward(self, node):

        if node is None:
            return

        h_a, c_a = node.parent.h
        assert h_a is not None, " node <{}> parent's state hasn't updated .".format(
            node.label)
        h_a, c_a = self.cell_a(node.parent.word_embed, (h_a, c_a))

        if node.left_brother is not None:
            h_f, c_f = node.left_brother.h
            assert h_f is not None, " node <{}> left sibling's state hasn't updated .".format(
                node.label)
            h_f, c_f = self.cell_f(node.left_brother.word_embed, (h_f, c_f))

        else:
            h_f, c_f = self.init_state(node.parent.h[0].size(0))

        h = F.tanh(self.linear_a(h_a) + self.linear_f(h_f))
        node.h = h, c_f
        node.word_embed = self.embed(node.word_idx)

        for c in [node.lc, node.mc, node.rc]:
            self._forward(c)

    def forward(self, tree, feature):
        bsz = feature.size(0)

        h_a, c_a = self.init_state(bsz)
        fc_embed = self.fc_embed(feature)
        h_a, c_a = self.cell_a(fc_embed, (h_a, c_a))

        h_f, c_f = self.init_state(bsz)
        h = F.tanh(self.linear_a(h_a) + self.linear_f(h_f))
        node = tree.root
        node.h = h, c_f
        node.word_embed = self.embed(node.word_idx)
        for c in [node.lc, node.mc, node.rc]:
            self._forward(c)

        # acce 
        target = [n.word_idx for n in tree.nodes.values()]
        target = torch.stack(target, dim=1)

        states = [n.h[0] for n in tree.nodes.values()]
        states = torch.stack(states, dim=1)
        output = self.logit(states)

        target = target.view(-1)
        output = output.view(-1, output.size(-1))
        return output, target

    def from_scratch(self, feature):
        h_a, c_a = self.init_state(1)
        word_embed = self.fc_embed(feature)
        h_a, c_a = self.cell_a(word_embed, (h_a, c_a))
        # h_a = feature
        h_f, c_f = self.init_state(1)
        h = F.tanh(self.linear_a(h_a) + self.linear_f(h_f))

        logits = self.logit(h)
        # print(logits.size())
        probs = F.softmax(logits, dim=1)
        return (h, c_f), probs

    def step(self, node):
        h_a, c_a = node.parent.h
        word_embed_a = self.embed(node.parent.word_idx)
        h_a, c_a = self.cell_a(word_embed_a, (h_a, c_a))

        if node.left_brother is not None:
            h_f, c_f = node.left_brother.h
            word_embed_f = self.embed(node.left_brother.word_idx)
            h_f, c_f = self.cell_f(word_embed_f, (h_f, c_f))
        else:
            h_f, c_f = self.init_state(1)

        h = F.tanh(self.linear_a(h_a) + self.linear_f(h_f))
        logits = self.logit(h)
        probs = F.softmax(logits, dim=1)
        return (h, c_f), probs

    def greedy_search(self, fc_feat, vocab, max_seq_length):
        (h_r, c_r), probs_r = self.from_scratch(fc_feat)
        _, word_idx_r = probs_r.max(dim=1)
        lex_r = vocab.idx2word[word_idx_r.item()]
        word_idx_r = torch.LongTensor([word_idx_r])

        node_r = TensorNode(Node(lex_r, 1, ""))
        node_r.word_idx = word_idx_r
        node_r.h = h_r, c_r

        tree = Tree()
        tree.rnn_type = 'LSTM'
        tree.root = node_r
        tree.nodes.update({node_r.idx - 1: node_r})

        tree.model = self
        tree.vocab = vocab
        tree.max_seq_length = max_seq_length
        tree.bfs_breed()
        return tree

    def beam_search(self, fc_feat, vocab, max_seq_length, beam_size):
        h_r, probs_r = self.from_scratch(fc_feat)
