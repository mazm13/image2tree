from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from queue import Queue
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.parameter import Parameter

import sys

sys.path.append('..')
from graph.node import TensorNode, Node
from graph.ternary import Tree


class MDLSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, att_feat_size, bias=True):
        super(MDLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_prob_lm = 0.5
        self.bias = bias

        self.weight_f1 = Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.weight_f2 = Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.weight_H = Parameter(torch.Tensor(2 * hidden_size, 2 * input_size + 2 * hidden_size))
        self.weight_I = Parameter(torch.Tensor(hidden_size, 2 * input_size + 2 * hidden_size + hidden_size))
        self.drop_out = nn.Dropout(self.drop_prob_lm)
        if bias:
            self.bias_f1 = Parameter(torch.Tensor(hidden_size))
            self.bias_f2 = Parameter(torch.Tensor(hidden_size))
            self.bias_H = Parameter(torch.Tensor(2 * hidden_size))
            self.bias_I = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_f1', None)
            self.register_parameter('bias_f2', None)
            self.register_parameter('bias_H', None)
            self.register_parameter('bias_I', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2, hx1, hx2, att_feats):
        self.check_forward_input(input1)
        self.check_forward_input(input2)
        self.check_forward_hidden(input1, hx1[0], '[0]')
        self.check_forward_hidden(input1, hx1[1], '[1]')
        self.check_forward_hidden(input2, hx2[0], '[0]')
        self.check_forward_hidden(input2, hx2[1], '[1]')

        H = torch.cat([input1, input2, hx1[0], hx2[0]], dim=1)
        gates = F.linear(H, self.weight_H, self.bias_H)
        ingate, outgate = gates.chunk(2, 1)

        ingate = F.sigmoid(ingate)
        outgate = F.sigmoid(outgate)

        H_hat = torch.cat([input1, input2, hx1[0], hx2[0], att_feats], dim=1)

        cellgate = F.linear(H_hat, self.weight_I, self.bias_I)
        cellgate = torch.tanh(cellgate)

        forgetgate1 = F.linear(torch.cat([input1, hx1[0]], dim=1), self.weight_f1, self.bias_f1)
        forgetgate2 = F.linear(torch.cat([input2, hx2[0]], dim=1), self.weight_f2, self.bias_f2)
        forgetgate1 = F.sigmoid(forgetgate1)
        forgetgate2 = F.sigmoid(forgetgate2)

        next_c = forgetgate1 * hx1[1] + forgetgate2 * hx2[1] + ingate * cellgate
        next_h = outgate * F.tanh(next_c)

        output = self.drop_out(next_h)
        return output, (next_h, next_c)


class TreeModelMDIN(nn.Module):
    def __init__(self, opt):
        super(TreeModelMDIN, self).__init__()
        self.opt = opt
        self.gpu = opt.use_cuda
        self.hidden_size = opt.hidden_size

        self.fc_embed = nn.Linear(opt.fc_feat_size, opt.embed_size)
        self.embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        self.md_lstm = MDLSTMCell(opt.embed_size, opt.hidden_size, opt.att_feat_size)
        self.logit = nn.Linear(opt.hidden_size, opt.vocab_size)

    def init_state(self, bsz):
        h = torch.zeros(bsz, self.opt.hidden_size, requires_grad=False)
        c = torch.zeros(bsz, self.opt.hidden_size, requires_grad=False)
        if self.gpu:
            h, c = h.cuda(), c.cuda()
        return h, c

    def init_input(self, bsz):
        zero_input = torch.zeros(bsz, self.opt.embed_size, requires_grad=False)
        if self.gpu:
            zero_input = zero_input.cuda()
        return zero_input

    def set_gpu(self, gpu):
        self.gpu = gpu

    def forward(self, word_idx, father_idx, fc_feats):
        batch_size = word_idx.size(0)
        fc_feats = self.fc_embed(fc_feats)

        h_states = torch.zeros(batch_size, word_idx.size(1), self.hidden_size, requires_grad=True).cuda()
        c_states = torch.zeros(batch_size, word_idx.size(1), self.hidden_size, requires_grad=True).cuda()
        outputs = []

        max_length = word_idx.size(1)
        for i in range(max_length):
            if i == 0:
                x_a = self.init_input(batch_size)
                x_f = self.init_input(batch_size)
                h_a = self.init_state(batch_size)
                h_f = self.init_state(batch_size)
            else:
                f_idx = father_idx[:, i].clone().unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                # print(word_idx.size())
                # print(father_idx)
                x_a = torch.gather(word_idx, dim=1, index=father_idx[:, i].clone().unsqueeze(1))
                x_a = self.embed(x_a.squeeze(dim=1))
                h_a_h = torch.gather(h_states, dim=1, index=f_idx).squeeze(dim=1)
                h_a_c = torch.gather(c_states, dim=1, index=f_idx).squeeze(dim=1)

                h_a = h_a_h, h_a_c
                if (i - 1) % 3 == 0:
                    # for those do not have left sibling
                    x_f = self.init_input(batch_size)
                    h_f = self.init_state(batch_size)
                else:
                    x_f = word_idx[:, i - 1].clone()
                    x_f = self.embed(x_f)
                    h_f_h = h_states[:, i - 1].clone()
                    h_f_c = c_states[:, i - 1].clone()
                    h_f = h_f_h, h_f_c

            output, (next_h, next_c) = self.md_lstm(x_a, x_f, h_a, h_f, fc_feats)
            h_states[:, i, :] = next_h
            c_states[:, i, :] = next_c
            outputs.append(output)

        outputs = self.logit(torch.stack(outputs, dim=1))
        outputs = F.log_softmax(outputs, dim=2)

        return outputs

    def from_scratch(self, fc_feats):
        ha, ca = self.init_state(1)
        hf, cf = self.init_state(1)
        input_f = self.init_input(1)
        input_a = self.init_input(1)

        output, states = self.md_lstm(fc_feats, input_f, (ha, ca), (hf, cf), fc_feats)
        logits = self.logit(output)
        logprobs = F.log_softmax(logits, dim=1)
        return states, logprobs

    def complete_node(self, node, vocab, fc_feats):
        h_a, c_a = node.parent.h
        word_embed_a = self.embed(node.parent.word_idx)
        if node.left_brother is not None:
            h_f, c_f = node.left_brother.h
            input_f = self.embed(node.left_brother.word_idx)
        else:
            h_f, c_f = self.init_state(1)
            input_f = self.init_input(1)

        output, states = self.md_lstm(word_embed_a, input_f, (h_a, c_a), (h_f, c_f), fc_feats)
        logits = self.logit(output)
        logprobs = F.log_softmax(logits, dim=1)

        logprob, word_idx = logprobs.max(dim=1)
        lex = vocab.idx2word[word_idx.item()]

        node.lex = lex
        node.label = node.lex + '-' + str(node.idx)
        node.word_idx = word_idx
        node.logprob = logprob.item()
        node.h = states

    def _sample(self, fc_feats, vocab, max_seq_length):
        fc_feats = self.fc_embed(fc_feats)

        states_r, probs_r = self.from_scratch(fc_feats)
        logprob, word_idx_r = probs_r.max(dim=1)
        lex_r = vocab.idx2word[word_idx_r.item()]
        word_idx_r = torch.LongTensor([word_idx_r])

        node_r = TensorNode(Node(lex_r, 1, ""))
        node_r.word_idx = word_idx_r
        node_r.h = states_r
        node_r.logprob = logprob.item()

        tree = Tree()
        tree.root = node_r
        tree.nodes.update({node_r.idx - 1: node_r})

        queue = Queue()
        queue.put(tree.root)
        print(tree.root.lex)
        while not queue.empty() and len(tree.nodes) <= max_seq_length:
            node = queue.get()

            if node.lex == '<EOB>':
                continue

            idx = len(tree.nodes) + 1
            lc = TensorNode(Node("", idx, ""))
            lc.parent = node
            self.complete_node(lc, vocab, fc_feats)
            node.lc = lc
            lc.depth = node.depth + 1

            mc = TensorNode(Node("", idx + 1, ""))
            mc.parent = node
            mc.left_brother = lc
            self.complete_node(mc, vocab, fc_feats)
            node.mc = mc
            mc.depth = node.depth + 1
            lc.right_brother = mc

            rc = TensorNode(Node("", idx + 2, ""))
            rc.parent = node
            rc.left_brother = mc
            self.complete_node(rc, vocab, fc_feats)
            node.rc = rc
            rc.depth = node.depth + 1
            mc.right_brother = rc

            tree.nodes.update({lc.idx - 1: lc})
            tree.nodes.update({mc.idx - 1: mc})
            tree.nodes.update({rc.idx - 1: rc})

            queue.put(lc)
            queue.put(mc)
            queue.put(rc)

        return tree

#
# if __name__ == '__main__':
#     model = MDLSTMCell(3, 4, 5)
#     model = model.cuda()
#     input1 = torch.randn(2, 3).cuda()
#     input2 = torch.randn(2, 3).cuda()
#     h1 = torch.randn(2, 4).cuda()
#     h2 = torch.randn(2, 4).cuda()
#     c1 = torch.randn(2, 4).cuda()
#     c2 = torch.randn(2, 4).cuda()
#     att_res = torch.randn(2, 5).cuda()
#
#     hy, cy = model(input1, input2, (h1, c1), (h2, c2), att_res)
#     print(hy)
#     print(cy)
