from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch

from .AttModel import AttModel
from .TreeModel import TreeModel
from .TreeModel_1 import TreeModel1
from .TreeModelMD import TreeModelMD
from .TreeModel_2 import TreeModel2
from .TreeModelAttMD import TreeModelAttMD
from .TreeModelMDSOB import TreeModelMDSOB
from .TreeModelMDIN import TreeModelMDIN
from .DRNN import DRNN


def setup(opt):
    if opt.caption_model == 'att_model':
        model = AttModel(opt)
    elif opt.caption_model == 'tree_model':
        model = TreeModel(opt)
    elif opt.caption_model == 'tree_model_1':
        model = TreeModel1(opt)
    elif opt.caption_model == 'tree_model_md':
        model = TreeModelMD(opt)
    elif opt.caption_model == 'tree_model_2':
        model = TreeModel2(opt)
    elif opt.caption_model == 'tree_model_md_att':
        model = TreeModelAttMD(opt)
    elif opt.caption_model == 'tree_model_md_sob':
        model = TreeModelMDSOB(opt)
    elif opt.caption_model == 'drnn':
        model = DRNN(opt)
    elif opt.caption_model == 'tree_model_md_in':
        model = TreeModelMDIN(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        print("load weights from {}".format(opt.start_from))

        assert os.path.isdir(os.path.join('checkpoint', opt.start_from)), " %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join('checkpoint', opt.start_from, "infos_" + opt.start_from + "_best.pkl")), \
            "infos.pkl file does not exist in path %s" % opt.start_from
        model.load_state_dict(torch.load(os.path.join('checkpoint',
                                                      opt.start_from, 'model_' + opt.start_from + '_best.pth')))

    return model
