from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.parameter import Parameter

from graph.node import TensorNode, Node
from graph.ternary import Tree
from queue import Queue


class MDLSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MDLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_prob_lm = 0.5
        self.bias = bias
        self.weight_f1 = Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.weight_f2 = Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.weight_H = Parameter(torch.Tensor(3 * hidden_size, 2 * input_size + 2 * hidden_size))
        if bias:
            self.bias_f1 = Parameter(torch.Tensor(hidden_size))
            self.bias_f2 = Parameter(torch.Tensor(hidden_size))
            self.bias_H = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_f1', None)
            self.register_parameter('bias_f2', None)
            self.register_parameter('bias_H', None)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2, hx1=None, hx2=None):
        self.check_forward_input(input1)
        self.check_forward_input(input2)
        self.check_forward_hidden(input1, hx1[0], '[0]')
        self.check_forward_hidden(input1, hx1[1], '[1]')
        self.check_forward_hidden(input2, hx2[0], '[0]')
        self.check_forward_hidden(input2, hx2[1], '[1]')

        H = torch.cat([input1, input2, hx1[0], hx2[0]], dim=1)
        gates = F.linear(H, self.weight_H, self.bias_H)
        ingate, cellgate, outgate = gates.chunk(3, 1)

        ingate = F.sigmoid(ingate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        forgetgate1 = F.linear(torch.cat([input1, hx1[0]], dim=1), self.weight_f1, self.bias_f1)
        forgetgate2 = F.linear(torch.cat([input2, hx2[0]], dim=1), self.weight_f2, self.bias_f2)
        forgetgate1 = F.sigmoid(forgetgate1)
        forgetgate2 = F.sigmoid(forgetgate2)

        next_c = forgetgate1 * hx1[1] + forgetgate2 * hx2[1] + ingate * cellgate
        next_h = outgate * F.tanh(next_c)

        output = self.dropout(next_h)
        return output, (next_h, next_c)


class DRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_a = nn.LSTMCell(input_size, hidden_size)
        self.gate_f = nn.LSTMCell(input_size, hidden_size)
        self.to_pred = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, xa, xf, ha, hf):
        """
        return
            output: vector for predicting next label
            next_a, states pair passed to its children
            next_f, states pair passed to its right sibling
        """
        h_ia = self.gate_a(xa, ha)
        h_if = self.gate_f(xf, hf)
        output = self.to_pred(torch.cat([h_ia[0], h_if[0]], dim=1))
        output = torch.tanh(output)
        return output, h_ia, h_if


class DRNN(nn.Module):
    def __init__(self, opt):
        super(DRNN, self).__init__()
        self.opt = opt
        self.max_length = 40
        self.gpu = opt.use_cuda
        self.hidden_size = opt.hidden_size

        self.fc_embed = nn.Linear(opt.fc_feat_size, opt.embed_size)
        self.embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        # self.md_lstm = MDLSTMCell(opt.embed_size, opt.hidden_size)
        self.cell = DRNNCell(opt.embed_size, opt.hidden_size)
        self.logit = nn.Linear(opt.hidden_size, opt.vocab_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

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
    #
    # def init_state(self, bsz):
    #     weight = next(self.parameters())
    #     return (weight.new_zeros(bsz, self.opt.hidden_size),
    #             weight.new_zeros(bsz, self.opt.hidden_size))
    #
    # def init_input(self, bsz):
    #     weight = next(self.parameters())
    #     return weight.new_zeros(bsz, self.opt.embed_size)

    def set_gpu(self, gpu):
        self.gpu = gpu

    def forward(self, word_idx, father_idx, fc_feats):
        batch_size = fc_feats.size(0)
        h_states = torch.zeros(batch_size, word_idx.size(1), self.hidden_size, requires_grad=True).cuda()
        c_states = torch.zeros(batch_size, word_idx.size(1), self.hidden_size, requires_grad=True).cuda()
        h_siblings = torch.zeros(batch_size, word_idx.size(1), self.hidden_size, requires_grad=True).cuda()
        c_siblings = torch.zeros(batch_size, word_idx.size(1), self.hidden_size, requires_grad=True).cuda()
        outputs = []

        for i in range(word_idx.size(1)):
            if i == 0:
                x_a = self.fc_embed(fc_feats)
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
                    h_f_h = h_siblings[:, i - 1].clone()
                    h_f_c = c_siblings[:, i - 1].clone()
                    h_f = h_f_h, h_f_c
            # output, (next_h, next_c) = self.md_lstm(x_a, x_f, h_a, h_f)
            output, (child_h, child_c), (bro_h, bro_c) = self.cell(x_a, x_f, h_a, h_f)
            h_states[:, i, :] = child_h
            c_states[:, i, :] = child_c
            h_siblings[:, i, :] = bro_h
            c_siblings[:, i, :] = bro_c
            outputs.append(output)

        outputs = self.logit(torch.stack(outputs, dim=1))
        outputs = F.log_softmax(outputs, dim=2)

        return outputs

    def from_scratch(self, feature):
        h_a, c_a = self.init_state(1)
        word_embed = self.fc_embed(feature)

        h_f, c_f = self.init_state(1)
        input_f = self.init_input(1)

        # hidden, cell = self.md_lstm(word_embed, input_f, (h_a, c_a), (h_f, c_f))
        output, state_a, state_f = self.cell(word_embed, input_f, (h_a, c_a), (h_f, c_f))
        logits = self.logit(output)
        # print(logits.size())
        logprobs = F.log_softmax(logits, dim=1)
        return state_a, state_f, logprobs

    def beam_from_scratch(self, feature):
        h_a, c_a = self.init_state(1)
        word_embed = self.fc_embed(feature)

        h_f, c_f = self.init_state(1)
        input_f = self.init_input(1)

        hidden, cell = self.md_lstm(word_embed, input_f, (h_a, c_a), (h_f, c_f))

        logits = self.logit(hidden)
        logprobs = F.log_softmax(logits, dim=1)
        return (hidden, cell), logprobs

    def step(self, node):
        h_a, c_a = node.parent.h
        word_embed_a = self.embed(node.parent.word_idx)
        if node.left_brother is not None:
            h_f, c_f = node.left_brother.h
            input_f = self.embed(node.left_brother.word_idx)
        else:
            h_f, c_f = self.init_state(1)
            input_f = self.init_input(1)
        hidden, cell = self.md_lstm(word_embed_a, input_f, (h_a, c_a), (h_f, c_f))
        logits = self.logit(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        return (hidden, cell), log_probs

    def greedy_search(self, fc_feat, vocab, max_seq_length):
        (h_r, c_r), probs_r = self.from_scratch(fc_feat)
        logprob, word_idx_r = probs_r.max(dim=1)
        lex_r = vocab.idx2word[word_idx_r.item()]
        word_idx_r = torch.LongTensor([word_idx_r])

        node_r = TensorNode(Node(lex_r, 1, ""))
        node_r.word_idx = word_idx_r
        node_r.h = h_r, c_r
        node_r.logprob = logprob.item()

        tree = Tree()
        tree.rnn_type = 'LSTM'
        tree.root = node_r
        tree.nodes.update({node_r.idx - 1: node_r})

        tree.model = self
        tree.vocab = vocab
        tree.max_seq_length = max_seq_length
        tree.bfs_breed()
        return tree

    def complete_node(self, node):
        state_a = node.parent.h
        xt_a = node.parent.word_embed
        if node.left_brother is not None:
            # state_f = node.left_brother.h
            state_f = node.left_brother.hf
            # xt_f = self.embed(node.left_brother.word_idx)
            xt_f = node.left_brother.word_embed
        else:
            state_f = self.init_state(1)
            xt_f = self.init_input(1)
        # output, state = self.md_lstm(xt_a, xt_f, state_a, state_f)
        output, state_a, state_f = self.cell(xt_a, xt_f, state_a, state_f)
        logits = self.logit(output)
        logprobs = F.log_softmax(logits, dim=1)
        logprob, word_idx = logprobs.max(dim=1)

        node.word_idx = word_idx
        node.logprob = logprob.item()
        # node.h = output, state
        node.h = state_a
        node.hf = state_f
        node.word_embed = self.embed(word_idx)

    def _sample(self, fc_feats, vocab, max_seq_length):
        state_a = self.init_state(1)
        xt_a = self.fc_embed(fc_feats)
        state_f = self.init_state(1)
        xt_f = self.init_input(1)

        # output, state = self.md_lstm(xt_a, xt_f, state_a, state_f)
        output, state_a, state_f = self.cell(xt_a, xt_f, state_a, state_f)
        logits = self.logit(output)
        logprobs = F.log_softmax(logits, dim=1)

        logprob, word_idx_r = logprobs.max(dim=1)
        word_idx_r = torch.LongTensor([word_idx_r])

        node_r = TensorNode(Node("", 1, ""))
        node_r.word_idx = word_idx_r
        node_r.word_embed = self.embed(word_idx_r)
        # node_r.h = output, state
        node_r.h = state_a
        node_r.hf = state_f
        node_r.logprob = logprob.item()

        tree = Tree()
        tree.root = node_r
        tree.nodes.update({node_r.idx - 1: node_r})

        eob_idx = vocab('<EOB>')

        queue = Queue()
        queue.put(tree.root)
        while not queue.empty() and len(tree.nodes) <= max_seq_length:
            node = queue.get()

            if node.word_idx.item() == eob_idx:
                continue

            idx = len(tree.nodes) + 1
            lc = TensorNode(Node("", idx, ""))
            lc.parent = node
            self.complete_node(lc)
            node.lc = lc
            lc.depth = node.depth + 1

            mc = TensorNode(Node("", idx + 1, ""))
            mc.parent = node
            mc.left_brother = lc
            self.complete_node(mc)
            node.mc = mc
            mc.depth = node.depth + 1
            lc.right_brother = mc

            rc = TensorNode(Node("", idx + 2, ""))
            rc.parent = node
            rc.left_brother = mc
            self.complete_node(rc)
            node.rc = rc
            rc.depth = node.depth + 1
            mc.right_brother = rc

            tree.nodes.update({lc.idx - 1: lc})
            tree.nodes.update({mc.idx - 1: mc})
            tree.nodes.update({rc.idx - 1: rc})

            queue.put(lc)
            queue.put(mc)
            queue.put(rc)

        for node in tree.nodes.values():
            node.lex = vocab.idx2word[node.word_idx.item()]
            node.label = node.lex + '-' + str(node.idx)
        return tree

    def beam_step(self, state_a, x_a, state_f=None, x_f=None):
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
        hidden, cell = self.md_lstm(word_embed_a, word_embed_f, (h_a, c_a), (h_f, c_f))
        logits = self.logit(hidden)
        logprobs = F.log_softmax(logits, dim=1)
        return (hidden, cell), logprobs

    def beam_search(self, fc_feat, vocab, max_seq_length, global_beam_size, local_beam_size,
                    depth_normalization_factor=0.8):

        (h_r, c_r), logprobs_r = self.from_scratch(fc_feat)
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
                                                        x_a=node.word_idx.clone())
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
                                                             x_f=local_candidates[c]['c'].clone())
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
                                                             x_f=m_local_candidates[c]['c'][-1])
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

    def get_logprobs_state(self, x_a, h_a, x_f=None, h_f=None):
        if h_a is None:
            h_a = self.init_state(1)
            word_embed_a = self.fc_embed(x_a)
        else:
            x_a = x_a.view(-1)
            word_embed_a = self.embed(x_a)
        if x_f is not None:
            assert h_f is not None
            x_f = x_f.view(-1)
            word_embed_f = self.embed(x_f)
        else:
            h_f = self.init_state(1)
            word_embed_f = self.init_input(1)
        hy, cy = self.md_lstm(word_embed_a, word_embed_f, h_a, h_f)
        logporbs = F.log_softmax(self.logit(hy), dim=1)
        return (hy, cy), logporbs

    def beam_test(self, fc_feat, vocab, global_bs, local_bs):
        h, logporb = self.get_logprobs_state(x_a=fc_feat, h_a=None)
        print(logporb.size())
        ys, ix = torch.topk(logporb, global_bs, dim=1)
        print(ys)
        print(ix)
        print([vocab.idx2word[ix[0][i].item()] for i in range(global_bs)])

        root_word = ix[0][0]
        _h, _logprob = self.get_logprobs_state(x_a=root_word, h_a=h)
        _ys, _ix = torch.topk(_logprob, local_bs, dim=1)
        print(_ys)
        print(_ix)
        print([vocab.idx2word[_ix[0][i].item()] for i in range(local_bs)])

        first_child = _ix[0][0]
        _h, _logprob = self.get_logprobs_state(x_a=root_word, h_a=h, x_f=first_child, h_f=_h)
        _ys, _ix = torch.topk(_logprob, local_bs, dim=1)
        print(_ys)
        print(_ix)
        print([vocab.idx2word[_ix[0][i].item()] for i in range(local_bs)])


def create_node(idx, lex, word_idx, h, depth):
    node = TensorNode(Node(lex, idx, ""))
    node.label = lex + '-' + str(idx)
    node.word_idx = word_idx.clone()
    node.h = (h[0].clone(), h[1].clone())
    node.depth = depth
    return node


def tuple_clone(t):
    return (t[0].clone(), t[1].clone())


if __name__ == '__main__':
    model = MDLSTMCell(3, 4)
    model = model.cuda()
    input1 = torch.randn(2, 3).cuda()
    input2 = torch.randn(2, 3).cuda()
    h1 = torch.randn(2, 4).cuda()
    h2 = torch.randn(2, 4).cuda()
    c1 = torch.randn(2, 4).cuda()
    c2 = torch.randn(2, 4).cuda()

    hy, cy = model(input1, input2, (h1, c1), (h2, c2))
    print(hy)
    print(cy)
