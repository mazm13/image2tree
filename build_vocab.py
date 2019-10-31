from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import pickle
import argparse
import json
from collections import Counter

from graph.graph import Graph


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<UNK>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(file_path, threshold):
    """Build a simple vocabulary wrapper."""
    with open(file_path, 'r') as f:
        dataset = json.load(f)

    counter = Counter()
    for i, sen in enumerate(dataset):
        dependecies = sen['depends']
        g = Graph()
        for dep in dependecies:
            gov_node = g.add_node(dep['governorGloss'], dep['governor'], "")
            dep_node = g.add_node(dep['dependentGloss'], dep['dependent'], "")
            g.add_edge(gov_node, dep_node, dep['dep'])
        caption = g.__str__()
        # print("{} # {}".format(i, caption))
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i + 1) % 1000 == 0:
            print("{} # {}".format(i + 1, caption))
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(dataset)))

    # If the word frequency is less than 'threshold', then the word is
    # discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    #   <bot>: begin of tree
    #   <eob>: end of branch
    vocab = Vocabulary()
    vocab.add_word('<ROOT>')
    vocab.add_word('<EOB>')
    vocab.add_word('<UNK>')
    vocab.add_word('<PAD>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(file_path=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

    for k, v in vocab.idx2word.items():
        print("{}: {}".format(k, v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='data/kar_train.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab_4.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=5,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
