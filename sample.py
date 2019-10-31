import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
import json
from torchvision import transforms as trn
from build_vocab import Vocabulary
import models
import skimage
import skimage.io
from config import cfg
from graph.graph import Graph
from graph.ternary import Tree
from graph.node import TensorNode, Node

from misc.resnet_utils import myResnet
import misc.resnet
from misc import utils

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for debugging
def load_from_tiny(idx):
    with open('data/small_dataset.json', 'r') as f:
        dataset = json.load(f)

    root = '/media/disk/mazm13/dataset/images/train2014'

    data = dataset[idx]
    img_id = str(data['img_id'])
    img_path = os.path.join(
        root, 'COCO_train2014_' + '0' * (12 - len(img_id)) + img_id + '.jpg')

    dependecies = data['dependency']
    g = Graph()
    for dep in dependecies:
        gov_node = g.add_node(dep['governorGloss'], dep['governor'], "")
        dep_node = g.add_node(dep['dependentGloss'], dep['dependent'], "")
        g.add_edge(gov_node, dep_node, dep['dep'])
    tree = g.to_tree()
    tree.padding()
    print("target sentence: " + tree.__str__())
    return img_path


def main(args):
    # Load vocabulary wrapper
    with open(cfg.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    cfg.vocab_size = len(vocab)

    if args.raw_image == "":
        img_path = load_from_tiny(args.idx)
    else:
        img_path = args.raw_image

    print(img_path)
    img = skimage.io.imread(img_path)
    img = img.astype('float32') / 255.0
    img = torch.from_numpy(img.transpose([2, 0, 1]))
    img = preprocess(img)
    img = img.cuda()

    # Build models
    my_resnet = getattr(misc.resnet, 'resnet101')()
    my_resnet.load_state_dict(torch.load(
        './data/imagenet_weights/resnet101.pth'))
    my_resnet = myResnet(my_resnet)
    my_resnet.eval()
    my_resnet.cuda()

    cfg.use_cuda = True
    cfg.id = args.id
    cfg.start_from = args.start_from
    cfg.max_seq_length = args.max_seq_length
    cfg.caption_model = args.caption_model
    decoder = models.setup(cfg)
    decoder.cuda()
    decoder.eval()

    with torch.no_grad():
        feature, att_feats = my_resnet(img)
    feature = feature.unsqueeze(0)
    att_feats = att_feats.view(-1, att_feats.size(-1))
    att_feats = att_feats.unsqueeze(0)
    print(att_feats.size())

    # tree = decoder.greedy_search(feature, vocab, args.max_seq_length)
    # init_state, init_logprobs = decoder.beam_from_scratch(feature)
    # candidates = decoder.beam_search(init_state, init_logprobs, 5, 5)
    # candidates = decoder.beam_test(feature, vocab, 5, 5)
    print("greedy search:\n")
    # word_idx, father_idx, mask = decoder._greedy_search(feature, att_feats, vocab, args.max_seq_length)
    word_idx, father_idx, mask, seqLogprobs = decoder._sample(feature, att_feats, args.max_seq_length)
    print(word_idx)
    print(father_idx)
    ratio = utils.seq2ratio(father_idx, mask)
    print(mask)
    mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
    print(mask)
    print(ratio)
    exit(0)

    words = [vocab.idx2word[word_idx[0][i].item()] for i in range(word_idx.size(1))]
    for i in range(word_idx.size(1)):
        print(i, word_idx[0][i].item(), words[i], father_idx[0][i].item(), mask[0][i].item())
    print(words)
    sents = utils.decode_sequence(vocab, word_idx, father_idx, mask)
    print(mask)
    print(sents)
    exit(0)
    logprob = [_.logprob for _ in tree.nodes.values()]
    logprob = reduce(lambda x, y: x + y, logprob)
    print("logprob: {}".format(logprob / len(tree.nodes) ** 0.8))
    print(tree.root.lex)
    print(tree.__str__())
    # with open('gs.dot', 'w') as f:
    #     f.write(tree.graphviz())
    # with open('gs_logprob.dot', 'w') as f:
    #     f.write(tree.graphviz_info())

    return

    print("\nbeam search:\n")
    candidates, completed_sentences = decoder.beam_search(feature, vocab, args.max_seq_length, args.global_beam_size,
                                                          args.local_beam_size)

    for cs in completed_sentences:
        cs.logprob /= len(cs.nodes) ** 2.0
    completed_sentences = sorted(completed_sentences, key=lambda x: -x.logprob)
    candidates = sorted(candidates, key=lambda x: -x.logprob)

    with open('bs_0.dot', 'w') as f:
        f.write(completed_sentences[0].graphviz())

    with open('bs_1.dot', 'w') as f:
        f.write(completed_sentences[1].graphviz())

    print("\ncompleted sentences:")
    for candidate in completed_sentences:
        print("logprob: {}, sentence: {}".format(candidate.logprob, candidate.__str__()))
        # comment here for checking
        # logprob = [_.logprob for _ in candidate.nodes.values()]
        # print("logprob from every node: {}".format(reduce(lambda x, y: x + y, logprob)))
    print("\npartial sentences:")
    for candidate in candidates:
        print("logprob: {}, sentence: {}".format(candidate.logprob, candidate.__str__()))

        #
        # with open('g.dot', 'w') as f:
        #     f.write(tree.graphviz())
        #
        # print(tree.__str__())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_length', type=int,
                        default=40, help='max sequence length')
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--raw_image', type=str, default="png/example.png")
    parser.add_argument('--id', type=str, default='',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--start_from', type=str, required=True,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                                'infos.pkl'         : configuration;
                                'checkpoint'        : paths to model file(s) (created by tf).
                                                      Note: this file contains absolute paths, be careful when moving files around;
                                'model.ckpt-*'      : file(s) with model definition (created by tf)
                            """)
    parser.add_argument('--caption_model', type=str, default='tree_model_md',
                        help="tree_model, att_model")
    parser.add_argument('--global_beam_size', type=int, default=5)
    parser.add_argument('--local_beam_size', type=int, default=5)
    args = parser.parse_args()
    main(args)
