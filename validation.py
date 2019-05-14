# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# validation.py
#
# Validation phase. Model tuning of attribute label embedding (ALE) method on APY dataset
# 15 seen classes   :   bird - cat - mug - bus - diningtable - bottle - car - boat
#                       dog - zebra - monkey - centaur - chair - bicycle - building
# 5 unseen classes  :   aeroplane - wolf - carriage - sofa - bag
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# April, 2019
# --------------------------------------------------
import random
random.seed(123)

import numpy as np
np.random.seed(123)

import torch
torch.manual_seed(123)

from easydict import EasyDict as edict
from tools import load_data, map_labels
from model import Network, evaluate
from config import OBJPATH
import pickle


def main():

    print('#####    VALIDATION PHASE    #####')

    __C = edict()

    with open(OBJPATH, 'rb') as infile:
        __C = pickle.load(infile)

    # ---------------------------------------------------------------------------------------------------------------- #

    allClassVectors = load_data(__C.ALL_CLASS_VEC,          'all_class_vec')

    trainFeatures   = load_data(__C.TRAIN_FEATURES,         'train_features')
    trainLabels     = load_data(__C.TRAIN_LABELS,           'train_labels')

    seenFeatures    = load_data(__C.VAL_SEEN_FEATURES,      'val_seen_features')
    seenLabels      = load_data(__C.VAL_SEEN_LABELS,        'val_seen_labels')

    unseenFeatures  = load_data(__C.VAL_UNSEEN_FEATURES,    'val_unseen_features')
    unseenLabels    = load_data(__C.VAL_UNSEEN_LABELS,      'val_unseen_labels')

    print("##" * 25)
    print("All Class Vectors        : ", allClassVectors.shape)

    print("Train Features           : ", trainFeatures.shape)
    print("Train Labels             : ", trainLabels.shape)

    print("Seen Features            : ", seenFeatures.shape)
    print("Seen Labels              : ", seenLabels.shape)

    print("Unseen Features          : ", unseenFeatures.shape)
    print("Unseen Labels            : ", unseenLabels.shape)
    print("##" * 25)

    # ---------------------------------------------------------------------------------------------------------------- #

    n_class, attr_dim   = allClassVectors.shape
    n_train, feat_dim   = trainFeatures.shape
    n_seen, _           = seenFeatures.shape
    n_unseen, _         = unseenFeatures.shape


    print("##" * 25)
    print("Number of Train samples  : ", n_train)
    print("Number of Seen samples   : ", n_seen)
    print("Number of Unseen samples : ", n_unseen)
    print("Number of Classes        : ", n_class)
    print("Vector Dim               : ", attr_dim)
    print("Feature Dim              : ", feat_dim)
    print("##" * 25)

    # ---------------------------------------------------------------------------------------------------------------- #

    seenClassIndices    = np.unique(trainLabels)
    unseenClassIndices  = np.unique(unseenLabels)

    m_trainLabels       = map_labels(trainLabels, n_class, seenClassIndices)

    m_seenLabels        = map_labels(seenLabels, n_class, seenClassIndices)
    m_genSeenLabels     = seenLabels.flatten()

    m_unseenLabels      = map_labels(unseenLabels, n_class, unseenClassIndices)
    m_genUnseenLabels   = unseenLabels.flatten()

    # ---------------------------------------------------------------------------------------------------------------- #

    n_epoch     = __C.N_EPOCH
    batch_size  = __C.BATCH_SIZE
    n_batch     = n_train // batch_size
    offset      = n_train - (batch_size * n_batch)
    lr          = __C.LR

    model       = Network(feature_dim=feat_dim, vector_dim=attr_dim)
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)   # <-- Optimizer
    criterion   = torch.nn.CrossEntropyLoss(reduction='sum')    # <-- Loss Function

    # ---------------------------------------------------------------------------------------------------------------- #

    seenVectors     = torch.from_numpy(allClassVectors[seenClassIndices, :]).float()
    unseenVectors   = torch.from_numpy(allClassVectors[unseenClassIndices, :]).float()
    allVectors      = torch.from_numpy(allClassVectors).float()

    print("##" * 25)
    print("Seen Vector shape        : ", tuple(seenVectors.size()))
    print("Unseen Vector shape      : ", tuple(unseenVectors.size()))
    print("All Vector shape         : ", tuple(allVectors.size()))
    print("##" * 25)

    x_train = torch.from_numpy(trainFeatures).float()
    y_train = torch.from_numpy(m_trainLabels).long()

    # **************************************************************************************************************** #
    # **************************************************************************************************************** #

    for epochID in range(n_epoch):

        model.train()       # <-- Train Mode On

        # ------------------- #
        #  TRAINING
        # ------------------- #

        runningTrainLoss    = 0.
        trainIndices        = torch.randperm(n_train)

        for batchID in range(n_batch + 1):

            batchTrainIndices   = trainIndices[(batchID * batch_size):((batchID + 1) * batch_size)]
            trainLoss           = 0.

            optimizer.zero_grad()

            for index in batchTrainIndices:

                x_sample = x_train[index:(index + 1)]
                y_sample = y_train[index:(index + 1)]

                y_out       = model(x_sample, seenVectors)
                trainLoss   += criterion(y_out, y_sample)

            trainLoss.backward()    # <-- calculate gradients
            optimizer.step()        # <-- update weights

            runningTrainLoss += trainLoss.item()

        # ------------------- #
        # PRINT LOSS
        # ------------------- #
        print("%s\tTrain Loss: %s" % (str(epochID + 1), str(runningTrainLoss / n_train)))


        if (epochID + 1) % __C.INFO_EPOCH == 0:

            print("##" * 25)

            model.eval()        # <-- Evaluation Mode On

            x_trainFeatures     = torch.from_numpy(trainFeatures).float()
            y_trainLabels       = torch.from_numpy(m_trainLabels).long()

            x_seenFeatures      = torch.from_numpy(seenFeatures).float()
            y_seenLabels        = torch.from_numpy(m_seenLabels).long()
            y_genSeenLabels     = torch.from_numpy(m_genSeenLabels).long()

            x_unseenFeatures    = torch.from_numpy(unseenFeatures).float()
            y_unseenLabels      = torch.from_numpy(m_unseenLabels).long()
            y_genUnseenLabels   = torch.from_numpy(m_genUnseenLabels).long()

            # ------------------------------------------------------- #
            # TRAIN ACCURACY
            y_out       = model(x_trainFeatures, seenVectors)
            y_out       = torch.argmax(y_out, dim=1)
            trainScore  = torch.sum(y_out == y_trainLabels).item()
            trainAcc    = trainScore / n_train
            print("Train acc              : %s" % str(trainAcc))
            # ------------------------------------------------------- #
            # * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
            # ------------------------------------------------------- #
            # ZERO-SHOT ACCURACY
            zslAcc      = evaluate( model   = model,
                                    x       = x_unseenFeatures,
                                    y       = y_unseenLabels,
                                    vec     = unseenVectors)
            print("Zero-Shot acc          : %s" % str(zslAcc))
            # ------------------------------------------------------- #
            # * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
            # ------------------------------------------------------- #
            # GENERALIZED SEEN ACCURACY
            gSeenAcc    = evaluate( model   = model,
                                    x       = x_seenFeatures,
                                    y       = y_genSeenLabels,
                                    vec     = allVectors)
            print("Generalized Seen acc   : %s" % str(gSeenAcc))
            # ------------------------------------------------------- #
            # * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
            # ------------------------------------------------------- #
            # GENERALIZED UNSEEN ACCURACY
            gUnseenAcc  = evaluate( model   = model,
                                    x       = x_unseenFeatures,
                                    y       = y_genUnseenLabels,
                                    vec     = allVectors)
            print("Generalized Unseen acc : %s" % str(gUnseenAcc))
            # ------------------------------------------------------- #
            # * ----- * ----- * ----- * ----- * ----- * ----- * ----- *
            # ------------------------------------------------------- #
            # GENERALIZED ZERO-SHOT ACCURACY
            hScore = (2 * gSeenAcc * gUnseenAcc) / (gSeenAcc + gUnseenAcc)
            print("H-Score                : %s" % str(hScore))
            # ------------------------------------------------------- #

            print("##" * 25)

    return

if __name__ == '__main__':
    main()
