from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from graph.graph import Graph
from graph.ternary import Tree
from graph.node import Node, TensorNode


class CocoDataset(Dataset):
    def __init__(self, opt, vocab):
        if opt.use_tiny:
            ann_path = opt.tiny_tree_path
        else:
            ann_path = opt.full_tree_path
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)
        self.vocab = vocab
        self.cocotalk_fc = opt.cocotalk_fc
        self.cocotalk_att = opt.cocotalk_att
        self.caption_model = opt.caption_model

    def __getitem__(self, item):
        data = self.ann[item]
        dependencies = data['depends']
        g = Graph()
        for dep in dependencies:
            gov_node = g.add_node(dep['governorGloss'], dep['governor'], "")
            dep_node = g.add_node(dep['dependentGloss'], dep['dependent'], "")
            g.add_edge(gov_node, dep_node, dep['dep'])
        tree = g.to_tree()
        tree.padding()

        for n in tree.nodes.values():
            word_idx = self.vocab(
                n.lex.lower() if n.lex != "<EOB>" else "<EOB>")
            n.word_idx = word_idx

        fc_feat = np.load(os.path.join(self.cocotalk_fc, str(data['img_id']) + '.npy'))
        if self.caption_model == 'att_model' or self.caption_model == 'tree_model_md_att':
            att_feat = np.load(os.path.join(self.cocotalk_att, str(data['img_id']) + '.npz'))['feat']
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            return fc_feat, att_feat, tree
        elif self.caption_model == 'tree_model' or self.caption_model == 'tree_model_1' \
                or self.caption_model == 'tree_model_md' or self.caption_model == 'tree_model_2':
            return fc_feat, tree
        else:
            raise Exception("Caption model not supported: {}".format(self.caption_model))

    def __len__(self):
        return len(self.ann)


def batch_tree_pad(nodes, guard, tree):
    # nodes and tree are in the same position
    lc = [n.lc if n is not None else None for n in nodes]
    mc = [n.mc if n is not None else None for n in nodes]
    rc = [n.rc if n is not None else None for n in nodes]

    if not any(lc):
        return

    idx = len(tree.nodes) + 1

    lex = [n.lex if n is not None else '<PAD>' for n in lc]
    lex = '(' + ' '.join(lex) + ')'
    word_idx = [n.word_idx if n is not None else 3 for n in lc]
    new_lc = TensorNode(Node(lex, idx, ""))
    new_lc.word_idx = torch.LongTensor(word_idx)
    guard.lc = new_lc

    lex = [n.lex if n is not None else '<PAD>' for n in mc]
    lex = '(' + ' '.join(lex) + ')'
    word_idx = [n.word_idx if n is not None else 3 for n in mc]
    new_mc = TensorNode(Node(lex, idx + 1, ""))
    new_mc.word_idx = torch.LongTensor(word_idx)
    guard.mc = new_mc

    lex = [n.lex if n is not None else '<PAD>' for n in rc]
    lex = '(' + ' '.join(lex) + ')'
    word_idx = [n.word_idx if n is not None else 3 for n in rc]
    new_rc = TensorNode(Node(lex, idx + 2, ""))
    new_rc.word_idx = torch.LongTensor(word_idx)
    guard.rc = new_rc

    tree.nodes.update({new_lc.idx: new_lc})
    tree.nodes.update({new_mc.idx: new_mc})
    tree.nodes.update({new_rc.idx: new_rc})

    batch_tree_pad(lc, new_lc, tree)
    batch_tree_pad(mc, new_mc, tree)
    batch_tree_pad(rc, new_rc, tree)


def collate_fn(batch):
    if len(batch[0]) == 3:
        fc_feat, att_feat, trees = zip(*batch)
        fc_feat, att_feat = torch.from_numpy(np.stack(fc_feat, axis=0)), torch.from_numpy(np.stack(att_feat, axis=0))
        feats = (fc_feat, att_feat)
    elif len(batch[0]) == 2:
        fc_feat, trees = zip(*batch)
        fc_feat = torch.from_numpy(np.stack(fc_feat, axis=0))
        feats = (fc_feat,)
    else:
        raise ValueError

    root_lex = [tree.root.lex for tree in trees]
    root_lex = '(' + ' '.join(root_lex) + ')'
    root_node = TensorNode(Node(root_lex, 1, ""))
    root_node.word_idx = torch.from_numpy(np.stack([tree.root.word_idx for tree in trees], axis=0))

    batch_tree = Tree()
    batch_tree.root = root_node
    batch_tree.nodes.update({0: root_node})

    batch_tree_pad([n.root for n in trees], root_node, batch_tree)
    batch_tree.update()

    return feats + (batch_tree,)


def get_coco_loader(opt, vocab, shuffle=True):
    coco_dataset = CocoDataset(opt, vocab)
    data_loader = DataLoader(dataset=coco_dataset,
                             batch_size=opt.batch_size,
                             shuffle=shuffle,
                             num_workers=opt.num_workers,
                             collate_fn=collate_fn)
    return data_loader


class CocoEvalDataset(Dataset):
    def __init__(self, kwargs):
        self.eval_split = kwargs['eval_split']
        with open(kwargs['dataset_coco'], 'r') as f:
            cocotalk = json.load(f)
        cocotalk = cocotalk['images']
        ann = [im for im in cocotalk if im['split'] == self.eval_split]
        self.ann = ann
        self.caption_model = kwargs['caption_model']
        self.cocotalk_fc = kwargs['cocotalk_fc']
        self.cocotalk_att = kwargs['cocotalk_att']

    def __getitem__(self, item):
        cocoid = self.ann[item]['cocoid']
        if self.caption_model == 'att_model' or self.caption_model == 'tree_model_md_att':
            fc_feat = np.load(os.path.join(self.cocotalk_fc, str(cocoid) + '.npy'))
            att_feat = np.load(os.path.join(self.cocotalk_att, str(cocoid) + '.npz'))['feat']
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            return cocoid, fc_feat, att_feat
        elif self.caption_model == 'tree_model' or self.caption_model == 'tree_model_1' \
                or self.caption_model == 'tree_model_md' or self.caption_model == 'tree_model_2' \
                or self.caption_model == 'tree_model_md_in' or self.caption_model == 'drnn':
            fc_feat = np.load(os.path.join(self.cocotalk_fc, str(cocoid) + '.npy'))
            return cocoid, fc_feat
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.ann)


def get_eval_loader(kwargs):
    coco_eval_dataset = CocoEvalDataset(kwargs)

    def fn(data):
        if len(data[0]) == 3:
            cocoid, fc_feat, att_feat = data[0]
            temp = [fc_feat, att_feat]
            fc_feat, att_feat = [torch.unsqueeze(torch.from_numpy(feat), dim=0) for feat in temp]
            return cocoid, fc_feat, att_feat
        elif len(data[0]) == 2:
            cocoid, fc_feat = data[0]
            fc_feat = torch.unsqueeze(torch.from_numpy(fc_feat), dim=0)
            return cocoid, fc_feat

    loader = DataLoader(dataset=coco_eval_dataset,
                        shuffle=False,
                        batch_size=50)
    return loader


if __name__ == '__main__':
    from config import cfg
    import pickle
    from build_vocab import Vocabulary
    from misc.utils import LanguageModelCriterion

    cfg.batch_size = 2
    cfg.use_att = 0
    cfg.use_tiny = 0
    cfg.caption_model = 'tree_model_md'

    print(cfg)

    with open(cfg.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    cfg.vocab_size = len(vocab)
    loader = get_coco_loader(cfg, vocab)

    depths = {}
    for data in loader:
        fc_feat, tree = data
        depth = tree.depth()
        if depth in depths:
            depths[depth] += 1
        else:
            depths[depth] = 1
        # target = [n.word_idx for n in tree.nodes.values()]
        # target = torch.stack(target, dim=1)
        # target = target.view(-1)

        # mask = torch.zeros(target.size(), dtype=torch.float)
        # for i in range(target.size(0)):
        #     mask[i] = 1 if mask[i].item() != 3 else 0

        # for i in range(target.size(0)):
        #    print(target[i].item(), mask[i].item())

        # break
    for k, v in depths.items():
        print("{}: {}".format(k, v))
