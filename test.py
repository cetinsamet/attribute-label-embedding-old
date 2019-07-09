# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# test.py
#
# Test phase. Training of attribute label embedding (ALE) method on APY dataset
# 20 seen classes   :   aeroplane - bicycle - bird - boat - bottle - bus - car - cat - chair - dog
#                       monkey - wolf - zebra - mug - building - bag - carriage - sofa - centaur - diningtable
# 12 unseen classes :   tvmonitor - goat - motorbike - cow - jetski - train - sheep - statue - horse - person
#                       pottedplant - donkey
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# March, 2019
# --------------------------------------------------
import random
random.seed(123)

import numpy as np
np.random.seed(123)

import torch
torch.manual_seed(123)

from torch.utils.data import TensorDataset, DataLoader
from easydict import EasyDict as edict
from tools import load_data, map_labels
from model import Network, evaluate
from config import OBJPATH
import pickle


def main():

    print('#####    TEST PHASE    #####')

    # read data
    __C = edict()

    with open(OBJPATH, 'rb') as infile:
        __C = pickle.load(infile)

    # ---------------------------------------------------------------------------------------------------------------- #

    #load data
    allClassVectors     = load_data(__C.ALL_CLASS_VEC,          'all_class_vec')

    trainvalFeatures    = load_data(__C.TRAINVAL_FEATURES,      'trainval_features')
    trainvalLabels      = load_data(__C.TRAINVAL_LABELS,        'trainval_labels')

    seenFeatures        = load_data(__C.TEST_SEEN_FEATURES,     'test_seen_features')
    seenLabels          = load_data(__C.TEST_SEEN_LABELS,       'test_seen_labels')

    unseenFeatures      = load_data(__C.TEST_UNSEEN_FEATURES,   'test_unseen_features')
    unseenLabels        = load_data(__C.TEST_UNSEEN_LABELS,     'test_unseen_labels')

    print("##" * 25)
    print("All Class Vectors            : ", allClassVectors.shape)

    print("TrainVal Features            : ", trainvalFeatures.shape)
    print("TrainVal Labels              : ", trainvalLabels.shape)

    print("Seen Features                : ", seenFeatures.shape)
    print("Seen Labels                  : ", seenLabels.shape)

    print("Unseen Features              : ", unseenFeatures.shape)
    print("Unseen Labels                : ", unseenLabels.shape)
    print("##" * 25)

    # ---------------------------------------------------------------------------------------------------------------- #

    # get data information
    n_class, attr_dim   = allClassVectors.shape
    n_train, feat_dim   = trainvalFeatures.shape
    n_seen, _           = seenFeatures.shape
    n_unseen, _         = unseenFeatures.shape

    print("##" * 25)
    print("Number of Classes            : ", n_class)
    print("Number of Train samples      : ", n_train)
    print("Number of Seen samples       : ", n_seen)
    print("Number of Unseen samples     : ", n_unseen)
    print("Attribute Dim                : ", attr_dim)
    print("Feature Dim                  : ", feat_dim)
    print("##" * 25)

    # --------------------------------------------------------------------------------------------------------------- #

    seenClassIndices    = np.unique(trainvalLabels)
    unseenClassIndices  = np.unique(unseenLabels)

    m_trainvalLabels    = map_labels(trainvalLabels, n_class, seenClassIndices)

    m_seenLabels        = map_labels(seenLabels, n_class, seenClassIndices)
    m_genSeenLabels     = seenLabels.flatten()

    m_unseenLabels      = map_labels(unseenLabels, n_class, unseenClassIndices)
    m_genUnseenLabels   = unseenLabels.flatten()

    # --------------------------------------------------------------------------------------------------------------- #

    # set network hyper-parameters
    n_epoch     = __C.N_EPOCH
    batch_size  = __C.BATCH_SIZE
    lr          = __C.LR

    # set network architecture, optimizer and loss function
    model       = Network(feature_dim=feat_dim, vector_dim=attr_dim)
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)   # <-- Optimizer
    criterion   = torch.nn.CrossEntropyLoss(reduction='sum')    # <-- Loss Function

    # --------------------------------------------------------------------------------------------------------------- #

    # convert data from numpy arrays to pytorch tensors
    x_trainvalFeatures  = torch.from_numpy(trainvalFeatures).float()
    y_trainvalLabels    = torch.from_numpy(m_trainvalLabels).long()

    x_seenFeatures      = torch.from_numpy(seenFeatures).float()
    y_seenLabels        = torch.from_numpy(m_seenLabels).long()
    y_genSeenLabels     = torch.from_numpy(m_genSeenLabels).long()

    x_unseenFeatures    = torch.from_numpy(unseenFeatures).float()
    y_unseenLabels      = torch.from_numpy(m_unseenLabels).long()
    y_genUnseenLabels   = torch.from_numpy(m_genUnseenLabels).long()

    seenVectors         = torch.from_numpy(allClassVectors[seenClassIndices, :]).float()
    unseenVectors       = torch.from_numpy(allClassVectors[unseenClassIndices, :]).float()
    allVectors          = torch.from_numpy(allClassVectors).float()

    print("##" * 25)
    print("Seen Vector shape            : ", tuple(seenVectors.size()))
    print("Unseen Vector shape          : ", tuple(unseenVectors.size()))
    print("All Vector shape             : ", tuple(allVectors.size()))
    print("##" * 25)

    # initialize data loader
    trainvalData    = TensorDataset(x_trainvalFeatures, y_trainvalLabels)
    trainvalLoader  = DataLoader(trainvalData, batch_size=batch_size, shuffle=True)

    # **************************************************************************************************************** #
    #                                            ATTRIBUTE LABEL EMBEDDING                                             #
    # **************************************************************************************************************** #

    max_zslAcc      = float('-inf')
    max_gSeenAcc    = float('-inf')
    max_gUnseenAcc  = float('-inf')
    max_hScore      = float('-inf')

    # -------------------- #
    #       TRAINING       #
    # -------------------- #

    for epochID in range(n_epoch):

        model.train()       # <-- Train Mode On
        running_trainval_loss = 0.

        for x, y in trainvalLoader:

            y_out           = model(x, seenVectors)
            trainval_loss   = criterion(y_out, y)

            optimizer.zero_grad()       # <-- set gradients to zero
            trainval_loss.backward()    # <-- calculate gradients
            optimizer.step()            # <-- update weights

            running_trainval_loss += trainval_loss.item()

        # ---------------------- #
        #       PRINT LOSS       #
        # ---------------------- #
        print("%s\tTrain Loss: %s" % (str(epochID + 1), str(running_trainval_loss / n_train)))

        if (epochID + 1) % __C.INFO_EPOCH == 0:

            # ---------------------- #
            #       EVALUATION       #
            # ---------------------- #

            model.eval()  # <-- Evaluation Mode On

            print("##" * 25)
            # ------------------------------------------------------- #
            # TRAIN ACCURACY
            y_out       = model(x_trainvalFeatures, seenVectors)
            y_out       = torch.argmax(y_out, dim=1)
            trainScore  = torch.sum(y_out == y_trainvalLabels).item()
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
            if gSeenAcc + gUnseenAcc == 0.:
                hScore = 0.
            else:
                hScore = (2 * gSeenAcc * gUnseenAcc) / (gSeenAcc + gUnseenAcc)
            print("H-Score                : %s" % str(hScore))
            # ------------------------------------------------------- #

            if hScore > max_hScore:
                max_zslAcc      = zslAcc
                max_gSeenAcc    = gSeenAcc
                max_gUnseenAcc  = gUnseenAcc
                max_hScore      = hScore
                torch.save(model, 'model/ale.pt')
                print("ALE is saved.")

            print("##" * 25)

    print("Zsl Acc: %.5s\tGen Seen Acc: %.5s\tGen Unseen Acc: %.5s\t\033[1mH-Score: %.5s\033[0m" \
          % (str(max_zslAcc), \
             str(max_gSeenAcc), \
             str(max_gUnseenAcc), \
             str(max_hScore)))

    return

if __name__ == '__main__':
    main()
