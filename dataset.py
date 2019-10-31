import os
import numpy as np
import torch
import h5py
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SeqtreeCOCO(Dataset):
    def __init__(self, opt=None):
        self.hdf5_path = 'data/trainset.h5'
        self.max_length = 40
        self.h5_label_file = h5py.File(self.hdf5_path, 'r', driver='core')
        self.cocotalk_fc = 'data/cocotalk_fc'
        self.cocotalk_att = 'data/cocotalk_att'
        self.use_att = 1
        self._size = self.h5_label_file['image_ids'].shape[0]

    def __getitem__(self, item):
        father_idx = np.zeros(self.max_length, dtype='int')
        father_idx[:] = self.h5_label_file['father_idx'][item, :]
        word_idx = np.zeros(self.max_length, dtype='int')
        word_idx[:] = self.h5_label_file['word_idx'][item, :]

        label_length = self.h5_label_file['label_length'][item]
        mask = np.zeros(self.max_length, dtype='float32')
        mask[:label_length] = 1.0

        image_id = self.h5_label_file['image_ids'][item]
        # print(image_id)
        fc_feats = np.load(os.path.join(self.cocotalk_fc,
                                        str(image_id) + '.npy'))
        data = {}
        data['word_idx'] = word_idx
        data['father_idx'] = father_idx
        data['masks'] = mask
        data['fc_feats'] = fc_feats
        data['image_id'] = image_id.astype('int')

        if self.use_att:
            att_feats = np.load(os.path.join(self.cocotalk_att, str(image_id) + '.npz'))['feat']
            att_feats = att_feats.reshape(-1, att_feats.shape[-1])
            data['att_feats'] = att_feats

        return data

    def __len__(self):
        return self._size


if __name__ == '__main__':
    seqtree_coco = SeqtreeCOCO()
    dataloader = DataLoader(seqtree_coco, batch_size=16, shuffle=True, num_workers=0)

    import time
    import pickle
    from build_vocab import Vocabulary

    with open('data/vocab_4.pkl', 'rb') as f:
        vocab = pickle.load(f)

    with open('data/idx2caps.json', 'r') as f:
        idx2caps = json.load(f)
    idx2caps = {int(k): v for k, v in idx2caps.items()}

    start = time.time()
    for i, data in enumerate(dataloader):
        # _temp = data['fc_feats']
        words = [vocab.idx2word[data['word_idx'][0][i].item()] for i in range(data['word_idx'].size(1))]
        print(words)
        # print(data['word_idx'])
        # print(data['father_idx'])
        # print(data['masks'])
        # print(data['image_id'])
        # print(data['fc_feats'])
        for j in range(16):
            print(idx2caps[data['image_id'][j].item()])

        break
        # print(data['fc_feats'].shape)
    print(time.time() - start)
