# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# demo_eval.py
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# April, 2019
# --------------------------------------------------
from config import MODEL_PATH, MAIN_DATAPATH, VAL_DATAPATH, TEST_DATAPATH
from tools import load_data, map_labels
from model import evaluate
import numpy as np
import torch


def main():

    allClassVectors     = load_data(MAIN_DATAPATH + '/all_class_vec.mat', 'all_class_vec')
    seenFeatures        = load_data(TEST_DATAPATH + '/test_seen_features.mat', 'test_seen_features')
    seenLabels          = load_data(TEST_DATAPATH + '/test_seen_labels.mat', 'test_seen_labels')
    unseenFeatures      = load_data(TEST_DATAPATH + '/test_unseen_features.mat', 'test_unseen_features')
    unseenLabels        = load_data(TEST_DATAPATH + '/test_unseen_labels.mat', 'test_unseen_labels')

    # ---------------------------------------------------------------------------------------------------------------- #

    n_class, _          = allClassVectors.shape

    seenClassIndices    = np.unique(seenLabels)
    unseenClassIndices  = np.unique(unseenLabels)

    m_seenLabels        = map_labels(seenLabels, n_class, seenClassIndices)
    m_genSeenLabels     = seenLabels.flatten()

    m_unseenLabels      = map_labels(unseenLabels, n_class, unseenClassIndices)
    m_genUnseenLabels   = unseenLabels.flatten()

    # ---------------------------------------------------------------------------------------------------------------- #

    unseenVectors       = torch.from_numpy(allClassVectors[unseenClassIndices, :]).float()
    allVectors          = torch.from_numpy(allClassVectors).float()

    # ---------------------------------------------------------------------------------------------------------------- #

    model               = torch.load(MODEL_PATH) # <--- Load model

    x_seenFeatures      = torch.from_numpy(seenFeatures).float()
    y_seenLabels        = torch.from_numpy(m_seenLabels).long()
    y_genSeenLabels     = torch.from_numpy(m_genSeenLabels).long()

    x_unseenFeatures    = torch.from_numpy(unseenFeatures).float()
    y_unseenLabels      = torch.from_numpy(m_unseenLabels).long()
    y_genUnseenLabels   = torch.from_numpy(m_genUnseenLabels).long()

    print("##" * 25)
    # ------------------------------------------------------- #
    # * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
    # ------------------------------------------------------- #
    # ZERO-SHOT ACCURACY
    zslAcc = evaluate(model = model,
                      x     = x_unseenFeatures,
                      y     = y_unseenLabels,
                      vec   = unseenVectors)
    print("Zero-Shot acc    : %s" % str(zslAcc))
    # ------------------------------------------------------- #
    # * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
    # ------------------------------------------------------- #
    # GENERALIZED SEEN ACCURACY
    gSeenAcc = evaluate(model   = model,
                        x       = x_seenFeatures,
                        y       = y_genSeenLabels,
                        vec     = allVectors)
    print("Gen Seen acc     : %s" % str(gSeenAcc))
    # ------------------------------------------------------- #
    # * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
    # ------------------------------------------------------- #
    # GENERALIZED UNSEEN ACCURACY
    gUnseenAcc = evaluate(model = model,
                          x     = x_unseenFeatures,
                          y     = y_genUnseenLabels,
                          vec   = allVectors)
    print("Gen Unseen acc   : %s" % str(gUnseenAcc))
    # ------------------------------------------------------- #
    # * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
    # ------------------------------------------------------- #
    # GENERALIZED ZERO-SHOT ACCURACY
    hScore = (2 * gSeenAcc * gUnseenAcc) / (gSeenAcc + gUnseenAcc)
    print("Harmonic Mean    : %s" % str(hScore))
    # ------------------------------------------------------- #
    print("##" * 25)

    return


if __name__ == '__main__':
    main()
