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
from misc import utils


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
        self.weight_I = Parameter(torch.Tensor(hidden_size, 2 * input_size + 2 * hidden_size + att_feat_size))
        # self.drop_out = nn.Dropout(self.drop_prob_lm)
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

        # output = self.drop_out(next_h)
        output = next_h
        return output, (next_h, next_c)


class TreeModelAttMDCore(nn.Module):
    def __init__(self, opt):
        super(TreeModelAttMDCore, self).__init__()
        self.hidden_size = opt.hidden_size
        self.att_size = opt.att_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.ha2att = nn.Linear(self.hidden_size, self.hidden_size)
        self.hf2att = nn.Linear(self.hidden_size, self.hidden_size)
        self.alpha_net = nn.Linear(self.hidden_size, 1)

    def forward(self, ha, hf, att_feats, p_att_feats):
        # The p_att_feats here is already projected
        att = p_att_feats.view(-1, self.att_size, self.att_hid_size)
        att_ha = self.ha2att(ha)
        att_hf = self.hf2att(hf)
        att_ha = att_ha.unsqueeze(1).expand_as(att)
        att_hf = att_hf.unsqueeze(1).expand_as(att)
        dot = att_ha + att_hf + att
        dot = torch.tanh(dot)
        dot = dot.view(-1, self.hidden_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, self.att_size)

        weight = F.softmax(dot, dim=1)
        att_feats_ = att_feats.view(-1, self.att_size, att_feats.size(-1))
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)

        return att_res


class TreeModelAttMD(nn.Module):
    def __init__(self, opt):
        super(TreeModelAttMD, self).__init__()
        self.opt = opt
        self.gpu = opt.use_cuda
        self.hidden_size = opt.hidden_size

        self.fc_embed = nn.Linear(opt.fc_feat_size, opt.embed_size)
        self.att_embed = nn.Linear(opt.att_feat_size, opt.embed_size)
        self.embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        self.md_lstm = MDLSTMCell(opt.embed_size, opt.hidden_size, opt.att_feat_size)
        self.logit = nn.Linear(opt.hidden_size, opt.vocab_size)
        self.attention = TreeModelAttMDCore(opt)

    def _prepare_feature(self, fc_feats, att_feats):
        fc_feats = self.fc_embed(fc_feats)
        p_att_feats = self.att_embed(att_feats)
        return fc_feats, att_feats, p_att_feats

    def init_state(self, bsz):
        h = torch.zeros(bsz, self.opt.hidden_size, requires_grad=False)
        c = torch.zeros(bsz, self.opt.hidden_size, requires_grad=False)
        # if self.gpu:
        h, c = h.cuda(), c.cuda()
        return h, c

    def init_input(self, bsz):
        zero_input = torch.zeros(bsz, self.opt.embed_size, requires_grad=False)
        # if self.gpu:
        #     zero_input = zero_input.cuda()
        zero_input = zero_input.cuda()
        return zero_input

    def set_gpu(self, gpu):
        self.gpu = gpu

    def forward(self, word_idx, father_idx, fc_feats, att_feats):
        batch_size = word_idx.size(0)
        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats)

        h_states = torch.zeros(batch_size, word_idx.size(1), self.hidden_size, requires_grad=True).cuda()
        c_states = torch.zeros(batch_size, word_idx.size(1), self.hidden_size, requires_grad=True).cuda()
        outputs = []

        max_length = word_idx.size(1)
        for i in range(max_length):
            if i == 0:
                x_a = fc_feats
                x_f = self.init_input(batch_size)
                h_a = self.init_state(batch_size)
                h_f = self.init_state(batch_size)
                att_res = att_feats.mean(dim=1)
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
                    # for those do not have left siblings
                    x_f = self.init_input(batch_size)
                    h_f = self.init_state(batch_size)
                else:
                    x_f = word_idx[:, i - 1].clone()
                    x_f = self.embed(x_f)
                    h_f_h = h_states[:, i - 1].clone()
                    h_f_c = c_states[:, i - 1].clone()
                    h_f = h_f_h, h_f_c

                att_res = self.attention(h_a[0], h_f[0], att_feats, p_att_feats)

            output, (next_h, next_c) = self.md_lstm(x_a, x_f, h_a, h_f, att_res)
            h_states[:, i, :] = next_h
            c_states[:, i, :] = next_c
            outputs.append(output)

        outputs = self.logit(torch.stack(outputs, dim=1))
        outputs = F.log_softmax(outputs, dim=2)

        return outputs

    def _greedy_search(self, fc_feats, att_feats, max_seq_length):
        batch_size = fc_feats.size(0)
        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats)
        h_states = torch.zeros(batch_size, max_seq_length, self.hidden_size, requires_grad=True).cuda()
        c_states = torch.zeros(batch_size, max_seq_length, self.hidden_size, requires_grad=True).cuda()

        word_idx = torch.zeros(batch_size, max_seq_length, requires_grad=True).long().cuda()
        father_idx = torch.zeros(batch_size, max_seq_length * 5, requires_grad=False).long().cuda()
        cursor = torch.LongTensor([range(1, 4) for _ in range(batch_size)]).cuda()
        mask = torch.ones(batch_size, max_seq_length, requires_grad=False).long().cuda()

        for i in range(0, max_seq_length):
            if i == 0:
                x_a = fc_feats
                x_f = self.init_input(batch_size)
                h_a = self.init_state(batch_size)
                h_f = self.init_state(batch_size)
                att_res = att_feats.mean(dim=1)
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
                    # for those do not have left siblings
                    x_f = self.init_input(batch_size)
                    h_f = self.init_state(batch_size)
                else:
                    x_f = word_idx[:, i - 1].clone()
                    x_f = self.embed(x_f)
                    h_f_h = h_states[:, i - 1].clone()
                    h_f_c = c_states[:, i - 1].clone()
                    h_f = h_f_h, h_f_c

                att_res = self.attention(h_a[0], h_f[0], att_feats, p_att_feats)

            output, (next_h, next_c) = self.md_lstm(x_a, x_f, h_a, h_f, att_res)
            h_states[:, i, :] = next_h
            c_states[:, i, :] = next_c

            output = self.logit(output)
            logprobs = F.log_softmax(output, dim=1)
            logprob, word = logprobs.max(dim=1)  # greedy search
            # important
            father_idx.scatter_(1, cursor, i)
            cursor[word != 1] += 3
            word_idx[:, i] = word

        # for i in range(max_seq_length):
        #     cond1 = torch.gather(mask, dim=1, index=father_idx[:, i].clone().unsqueeze(1))
        #     cond2 = torch.gather(word_idx, dim=1, index=father_idx[:, i].clone().unsqueeze(1))
        #     cond1 = cond1.squeeze(dim=1)
        #     cond2 = cond2.squeeze(dim=1)
        #     mask[cond1==0 or cond2==1, i] = 0
        mask = utils.seq2mask(word_idx)
        return word_idx, father_idx, mask

    def _sample(self, fc_feats, att_feats, max_seq_length):
        batch_size = fc_feats.size(0)
        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats)
        h_states = torch.zeros(batch_size, max_seq_length, self.hidden_size, requires_grad=True).cuda()
        c_states = torch.zeros(batch_size, max_seq_length, self.hidden_size, requires_grad=True).cuda()

        word_idx = torch.zeros(batch_size, max_seq_length, requires_grad=False).long().cuda()
        father_idx = torch.zeros(batch_size, max_seq_length * 5, requires_grad=False).long().cuda()
        seqLogprobs = torch.zeros(batch_size, max_seq_length, requires_grad=False).float().cuda()
        cursor = torch.LongTensor([range(1, 4) for _ in range(batch_size)]).cuda()

        for i in range(0, max_seq_length):
            if i == 0:
                x_a = fc_feats
                x_f = self.init_input(batch_size)
                h_a = self.init_state(batch_size)
                h_f = self.init_state(batch_size)
                att_res = att_feats.mean(dim=1)
            else:
                f_idx = father_idx[:, i].clone().unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                x_a = torch.gather(word_idx, dim=1, index=father_idx[:, i].clone().unsqueeze(1))
                x_a = self.embed(x_a.squeeze(dim=1))
                h_a_h = torch.gather(h_states, dim=1, index=f_idx).squeeze(dim=1)
                h_a_c = torch.gather(c_states, dim=1, index=f_idx).squeeze(dim=1)

                h_a = h_a_h, h_a_c
                if (i - 1) % 3 == 0:
                    # for those do not have left siblings
                    x_f = self.init_input(batch_size)
                    h_f = self.init_state(batch_size)
                else:
                    x_f = word_idx[:, i - 1].clone()
                    x_f = self.embed(x_f)
                    h_f_h = h_states[:, i - 1].clone()
                    h_f_c = c_states[:, i - 1].clone()
                    h_f = h_f_h, h_f_c

                att_res = self.attention(h_a[0], h_f[0], att_feats, p_att_feats)

            output, (next_h, next_c) = self.md_lstm(x_a, x_f, h_a, h_f, att_res)
            h_states[:, i, :] = next_h
            c_states[:, i, :] = next_c

            output = self.logit(output)
            logprobs = F.log_softmax(output, dim=1)
            probs = torch.exp(logprobs.data).cpu()

            word = torch.multinomial(probs, 1).cuda()
            sampleLogprobs = torch.gather(logprobs, dim=1, index=word)
            word = word.squeeze(dim=1).long()
            # important
            father_idx.scatter_(1, cursor, i)
            cursor[word != 1] += 3
            word_idx[:, i] = word.long()
            seqLogprobs[:, i] = sampleLogprobs.squeeze(dim=1)

        mask = utils.seq2mask(word_idx)
        return word_idx, father_idx, mask, seqLogprobs

    def from_scratch(self, fc_feats, att_feats):
        ha, ca = self.init_state(1)
        hf, cf = self.init_state(1)
        input_f = self.init_input(1)

        att_res = att_feats.mean(dim=1)
        output, states = self.md_lstm(fc_feats, input_f, (ha, ca), (hf, cf), att_res)
        logits = self.logit(output)
        logprobs = F.log_softmax(logits, dim=1)
        return states, logprobs

    def complete_node(self, node, vocab, att_feats, p_att_feats):
        h_a, c_a = node.parent.h
        word_embed_a = self.embed(node.parent.word_idx)
        if node.left_brother is not None:
            h_f, c_f = node.left_brother.h
            input_f = self.embed(node.left_brother.word_idx)
        else:
            h_f, c_f = self.init_state(1)
            input_f = self.init_input(1)
        att_res = self.attention(h_a, h_f, att_feats, p_att_feats)
        output, states = self.md_lstm(word_embed_a, input_f, (h_a, c_a), (h_f, c_f), att_res)
        logits = self.logit(output)
        logprobs = F.log_softmax(logits, dim=1)

        logprob, word_idx = logprobs.max(dim=1)
        lex = vocab.idx2word[word_idx.item()]

        node.lex = lex
        node.label = node.lex + '-' + str(node.idx)
        node.word_idx = word_idx
        node.logprob = logprob.item()
        node.h = states

    def greedy_search(self, fc_feats, att_feats, vocab, max_seq_length):
        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats)

        states_r, probs_r = self.from_scratch(fc_feats, att_feats)
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
        # print(tree.root.lex)
        while not queue.empty() and len(tree.nodes) <= max_seq_length:
            node = queue.get()

            if node.lex == '<EOB>':
                continue

            idx = len(tree.nodes) + 1
            lc = TensorNode(Node("", idx, ""))
            lc.parent = node
            self.complete_node(lc, vocab, att_feats, p_att_feats)
            node.lc = lc
            lc.depth = node.depth + 1

            mc = TensorNode(Node("", idx + 1, ""))
            mc.parent = node
            mc.left_brother = lc
            self.complete_node(mc, vocab, att_feats, p_att_feats)
            node.mc = mc
            mc.depth = node.depth + 1
            lc.right_brother = mc

            rc = TensorNode(Node("", idx + 2, ""))
            rc.parent = node
            rc.left_brother = mc
            self.complete_node(rc, vocab, att_feats, p_att_feats)
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

    def beam_step(self, state_a, x_a, state_f, x_f, att_feats, p_att_feats):
        # x_a size() with ()
        x_a = x_a.view(-1)
        word_embed_a = self.embed(x_a)
        if state_f is not None:
            assert x_f is not None
            h_f, c_f = state_f
            x_f = x_f.view(-1)
            word_embed_f = self.embed(x_f)
        else:
            h_f, c_f = self.init_state(1)
            word_embed_f = self.init_input(1)
        h_a, c_a = state_a
        att_res = self.attention(h_a, h_f, att_feats, p_att_feats)
        output, (hidden, cell) = self.md_lstm(word_embed_a, word_embed_f, (h_a, c_a), (h_f, c_f), att_res)
        logits = self.logit(hidden)
        logprobs = F.log_softmax(logits, dim=1)
        return (hidden, cell), logprobs

    def beam_search(self, fc_feat, att_feats, vocab, max_seq_length, global_beam_size, local_beam_size,
                    depth_normalization_factor=0.8):
        fc_feat, att_feats, p_att_feats = self._prepare_feature(fc_feat, att_feats)
        (h_r, c_r), logprobs_r = self.from_scratch(fc_feat, att_feats)
        ys, ix = torch.sort(logprobs_r, 1, True)
        candidates = []
        for c in range(global_beam_size):
            local_logprob = ys[0, c].item()
            candidates.append({'c': ix[0, c], 'p': local_logprob})
        candidate_trees = []

        for c in range(global_beam_size):
            lex = vocab.idx2word[candidates[c]['c'].item()]
            node = TensorNode(Node(lex, 1, ""))
            node.word_idx = candidates[c]['c']
            node.h = h_r.clone(), c_r.clone()
            node.logprob = candidates[c]['p']
            tree = Tree()
            tree.root = node
            tree.logprob = candidates[c]['p']
            tree.nodes.update({node.idx - 1: node})
            candidate_trees.append(tree)

        # for t in range((max_seq_length - 1) // 3):
        completed_sentences = []

        # for t in range(10):
        while True:
            new_candidates = []
            for tree_idx, tree in enumerate(candidate_trees):
                # print(tree.__str__())
                # print([_node.lex for _node in tree.nodes.values()])
                dead_tree = True
                for node in tree.nodes.values():
                    if node.lc is None and node.lex != '<EOB>':
                        dead_tree = False
                if dead_tree:
                    temp_candidate_dict = {'p': tree.logprob, 'idx': tree_idx, 'dead_tree': True}
                    new_candidates.append(temp_candidate_dict)
                    print("done sentence: {}".format(tree.__str__()))
                    continue

                new_candidates_per_tree = []
                for node in tree.nodes.values():

                    current_depth = node.depth

                    # find L(local size) groups children of this node via chain beam search
                    if node.lc is not None or node.lex == '<EOB>':
                        continue
                    local_candidates = []

                    (hs, cs), logprobs = self.beam_step(state_a=(node.h[0].clone(), node.h[1].clone()),
                                                        x_a=node.word_idx.clone(),
                                                        state_f=None,
                                                        x_f=None,
                                                        att_feats=att_feats, p_att_feats=p_att_feats)
                    ys, ix = torch.sort(logprobs, 1, True)
                    for c in range(local_beam_size):
                        local_logprob = ys[0, c].item()
                        local_candidates.append({'c': ix[0, c],
                                                 'p': local_logprob,
                                                 'seq_p': [local_logprob]})

                    # print("input: father ({}), left sibling (None)".format(node.lex))
                    # print("output candidates: {}".format([vocab.idx2word[cdd['c'].item()] for cdd in local_candidates]))

                    m_local_candidates = []
                    for c in range(local_beam_size):
                        (_h, _c), _logprobs = self.beam_step(state_a=(node.h[0].clone(), node.h[1].clone()),
                                                             x_a=node.word_idx.clone(),
                                                             state_f=(hs.clone(), cs.clone()),
                                                             x_f=local_candidates[c]['c'].clone(),
                                                             att_feats=att_feats,
                                                             p_att_feats=p_att_feats)
                        _ys, _ix = torch.sort(_logprobs, 1, True)
                        for q in range(local_beam_size):
                            local_logprob = _ys[0, q].item()
                            entry = {'c': [local_candidates[c]['c'], _ix[0, q]],
                                     'p': local_logprob + local_candidates[c]['p'],
                                     'seq_p': local_candidates[c]['seq_p'] + [local_logprob],
                                     'state': (_h, _c)}
                            m_local_candidates.append(entry)

                            # print("input: father ({}), left sibling ({})".format(node.lex, vocab.idx2word[local_candidates[c]['c'].item()]))
                            # print("output candidates: {}".format(
                            #     [vocab.idx2word[cdd['c'][1].item()] for cdd in m_local_candidates]))

                    m_local_candidates = sorted(m_local_candidates, key=lambda x: -x['p'])
                    m_local_candidates = m_local_candidates[:local_beam_size]

                    # print([{'c': cdd['c'], 'p': cdd['p'], 'state_len': len(cdd['state']), 'state_0_len': len(cdd['state'][0])} for cdd in m_local_candidates])

                    r_local_candidates = []
                    for c in range(local_beam_size):
                        f_state = (m_local_candidates[c]['state'][0].clone(), m_local_candidates[c]['state'][1].clone())
                        (_h, _c), _logprobs = self.beam_step(state_a=(node.h[0].clone(), node.h[1].clone()),
                                                             x_a=node.word_idx.clone(),
                                                             state_f=f_state,
                                                             x_f=m_local_candidates[c]['c'][-1],
                                                             att_feats=att_feats,
                                                             p_att_feats=p_att_feats)
                        _ys, _ix = torch.sort(_logprobs, 1, True)
                        for q in range(local_beam_size):
                            local_logprob = _ys[0, q].item()
                            entry = {'c': [_.clone() for _ in m_local_candidates[c]['c']],
                                     'p': local_logprob + m_local_candidates[c]['p'],
                                     'seq_p': m_local_candidates[c]['seq_p'] + [local_logprob],
                                     'state': [(m_local_candidates[c]['state'][0].clone(),
                                                m_local_candidates[c]['state'][1].clone()),
                                               (_h.clone(), _c.clone())]}
                            entry['c'].append(_ix[0, q].clone())
                            r_local_candidates.append(entry)

                    r_local_candidates = sorted(r_local_candidates, key=lambda x: -x['p'])
                    r_local_candidates = r_local_candidates[:local_beam_size]

                    # print([{'c': cdd['c'], 'p': cdd['p'], 'state_len': len(cdd['state']), 'state_0_len': len(cdd['state'][0])} for cdd in r_local_candidates])

                    for candidate in r_local_candidates:
                        candidate['state'].insert(0, (hs, cs))
                        # this should be proceed after selecting top-L's combination
                        # candidate['p'] += tree.logprob
                        # candidate['idx'] = tree_idx
                        candidate['fid'] = node.idx
                        # candidate['dead_tree'] = False

                    new_candidates_per_tree.append(r_local_candidates)

                # Now we get a list with length of (len(nodes) - len(<EOB>))
                # And the number of combination is local_batch_size ** length

                # inefficient but accurate method, exhaustivity
                # still using beam search

                combination_candidates = []
                for i in range(len(new_candidates_per_tree)):
                    if i == 0:
                        for j in range(local_beam_size):
                            combination_candidates.append({'p': new_candidates_per_tree[i][j]['p'], 'seq': [j]})
                        continue
                    new_combination_candidates = []
                    for p in range(local_beam_size):
                        for q in range(local_beam_size):
                            local_combination_logprob = new_candidates_per_tree[i][q]['p'] + combination_candidates[p][
                                'p']
                            local_seq = combination_candidates[p]['seq'] + [q]
                            new_combination_candidates.append({'p': local_combination_logprob, 'seq': local_seq})
                    new_combination_candidates = sorted(new_combination_candidates, key=lambda x: -x['p'])
                    new_combination_candidates = new_combination_candidates[:local_beam_size]
                    combination_candidates = new_combination_candidates

                # print(len(combination_candidates))

                done_combination_candidates = []
                for i in range(local_beam_size):
                    done_combination_candidates.append({'p': combination_candidates[i]['p'] + tree.logprob,
                                                        'combination': [new_candidates_per_tree[i][wi] for i, wi in
                                                                        enumerate(combination_candidates[i]['seq'])],
                                                        'dead_tree': False,
                                                        'idx': tree_idx})

                new_candidates.extend(done_combination_candidates)

            # print(len(new_candidates))
            new_candidates = sorted(new_candidates, key=lambda x: -x['p'])
            new_candidates = new_candidates[:global_beam_size]

            new_candidate_trees = []
            for _candidate in new_candidates:
                if _candidate['dead_tree']:
                    new_candidate_trees.append(candidate_trees[tree_idx].clone())
                    continue

                tree_idx = _candidate['idx']
                new_tree = candidate_trees[tree_idx].clone()

                for candidate in _candidate['combination']:
                    f_node = new_tree.nodes[candidate['fid'] - 1]

                    # print(f_node.__repr__())
                    idx = len(new_tree.nodes) + 1
                    lex = [vocab.idx2word[_.item()] for _ in candidate['c']]
                    seq_p = candidate['seq_p']

                    lc = TensorNode(Node(lex[0], idx, ""))
                    lc.parent = f_node
                    lc.word_idx = candidate['c'][0].clone()
                    f_node.lc = lc
                    lc.depth = f_node.depth + 1
                    lc.h = tuple_clone(candidate['state'][0])
                    lc.logprob = seq_p[0]

                    mc = TensorNode(Node(lex[1], idx + 1, ""))
                    mc.parent = f_node
                    mc.left_brother = lc
                    mc.word_idx = candidate['c'][1].clone()
                    f_node.mc = mc
                    mc.depth = f_node.depth + 1
                    mc.h = tuple_clone(candidate['state'][1])
                    lc.right_brother = mc
                    mc.logprob = seq_p[1]

                    rc = TensorNode(Node(lex[2], idx + 2, ""))
                    rc.parent = f_node
                    rc.left_brother = mc
                    rc.word_idx = candidate['c'][2].clone()
                    f_node.rc = rc
                    rc.depth = f_node.depth + 1
                    rc.h = tuple_clone(candidate['state'][2])
                    mc.right_brother = rc
                    rc.logprob = seq_p[2]

                    new_tree.nodes.update({lc.idx - 1: lc})
                    new_tree.nodes.update({mc.idx - 1: mc})
                    new_tree.nodes.update({rc.idx - 1: rc})

                    new_tree.logprob += candidate['p']

                # with open('beam_search.dot', 'w') as f:
                #     f.write(new_tree.graphviz())
                #
                # exit(0)
                dead_tree = True
                for node in new_tree.nodes.values():
                    if node.lc is None and node.lex != '<EOB>':
                        dead_tree = False

                if dead_tree:
                    # new_tree.logprob /= len(new_tree.nodes)**3.0
                    completed_sentences.append(new_tree)
                else:
                    new_candidate_trees.append(new_tree)

            candidate_trees = new_candidate_trees

            # if len(candidate_trees) == 0:
            #     break
            break_flag = True
            for tree in candidate_trees:
                if len(tree.nodes) < max_seq_length:
                    break_flag = False
            if break_flag:
                break

        return candidate_trees, completed_sentences


def tuple_clone(t):
    return (t[0].clone(), t[1].clone())

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
