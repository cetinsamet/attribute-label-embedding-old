# --------------------------------------------------
#
# ale.py
#
# Attribute Label Embedding (ALE) Method
# Written by cetinsamet
# March, 2019
# --------------------------------------------------

import numpy as np
np.random.seed(123)

import scipy.io as sio
import pickle

import torch
from torch import nn
from torch.autograd import Variable

from model import Network


def loadData(data, dataName):

    dataContent = sio.loadmat(data)
    dataContent = dataContent[dataName]
    return dataContent

def dense_to_one_hot(labels_dense, num_classes):

  num_labels        = labels_dense.shape[0]
  index_offset      = np.arange(num_labels) * num_classes
  labels_one_hot    = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def main():

    # ---------------------------------------------------------------------------------------------------------------- #

    allVectors      = loadData('data/AWA2P/side/attribute/allVectors.mat',  'allVectors')
    testVectors     = loadData('data/AWA2P/side/attribute/testVectors.mat', 'testVectors')

    trainFeatures   = loadData('data/AWA2P/features/res101/trainValFeatures.mat', 'trainValFeatures')
    trainLabels     = loadData('data/AWA2P/trainValLabels.mat', 'trainValLabels')

    testFeatures    = loadData('data/AWA2P/features/res101/testSeenFeatures.mat', 'testSeenFeatures')
    testLabels      = loadData('data/AWA2P/testSeenLabels.mat', 'testSeenLabels')

    unseenFeatures  = loadData('data/AWA2P/features/res101/testUnseenFeatures.mat', 'testUnseenFeatures')
    unseenLabels    = loadData('data/AWA2P/testUnseenLabels.mat',   'testUnseenLabels')

    # ---------------------------------------------------------------------------------------------------------------- #

    n_class, vec_dim    = allVectors.shape
    n_train, feat_dim   = trainFeatures.shape
    n_test, _           = testFeatures.shape
    n_unseen, _         = unseenFeatures.shape

    # ---------------------------------------------------------------------------------------------------------------- #

    seenClassIndices        = np.unique(trainLabels)
    unseenClassIndices      = np.unique(unseenLabels)

    trainOneHotLabels       = dense_to_one_hot(trainLabels, n_class)[:, seenClassIndices]

    genTestOneHotLabels     = dense_to_one_hot(testLabels, n_class)
    testOneHotLabels        = dense_to_one_hot(testLabels, n_class)[:, seenClassIndices]

    genUnseenOneHotLabels   = dense_to_one_hot(unseenLabels, n_class)
    unseenOneHotLabels      = dense_to_one_hot(unseenLabels, n_class)[:, unseenClassIndices]

    # ---------------------------------------------------------------------------------------------------------------- #

    n_epoch     = 2000
    batchSize   = 128
    n_batch     = n_train // batchSize
    learnRate   = 1e-5

    model       = Network(feature_dim=feat_dim, vector_dim=vec_dim, n_hidden=vec_dim)
    optimizer   = torch.optim.Adam(model.parameters())
    criterion   = nn.MultiMarginLoss(reduction='sum')

    # ---------------------------------------------------------------------------------------------------------------- #

    seenVectors     = torch.from_numpy(allVectors[seenClassIndices, :]).float()
    unseenVectors   = torch.from_numpy(allVectors[unseenClassIndices, :]).float()
    allVectors      = torch.from_numpy(allVectors).float()

    for epochID in range(n_epoch):

        model.train()

        runningLoss     = 0.
        trainIndices    = np.random.permutation(np.arange(n_train))

        for batchID in range(n_batch + 1):

            batchTrainIndices = trainIndices[(batchID * batchSize):((batchID + 1) * batchSize)]

            x   = torch.from_numpy(trainFeatures[batchTrainIndices, :]).float()
            y   = torch.from_numpy(np.squeeze(np.argmax(trainOneHotLabels[batchTrainIndices], axis=1))).long()

            optimizer.zero_grad()

            y_out   = model(x, seenVectors)
            loss    = criterion(y_out, y)

            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

        if ((epochID + 1) % 5) == 0:
            print("%s\tLoss: %s" % (str(epochID + 1), str(runningLoss / n_train)))

        if ((epochID + 1) % 25) == 0:

            model.eval()

            # -------------------------------------------------------------------------------------------------------- #
            # TRAINING ACCURACY

            x   = torch.from_numpy(trainFeatures).float()
            gt  = torch.from_numpy(np.squeeze(np.argmax(trainOneHotLabels, axis=1)))

            y_out = model(x, seenVectors)
            y_out = torch.argmax(y_out, dim=1)

            trainScore  = torch.sum(y_out == gt).item()
            trainAcc    = trainScore / n_train
            print("##" * 25)
            print("Training Acc       : %s" % str(trainAcc))
            # -------------------------------------------------------------------------------------------------------- #


            # -------------------------------------------------------------------------------------------------------- #
            # ZERO-SHOT ACCURACY

            x   = torch.from_numpy(unseenFeatures).float()
            gt  = torch.from_numpy(np.squeeze(np.argmax(unseenOneHotLabels, axis=1)))

            y_out = model(x, unseenVectors)
            y_out = torch.argmax(y_out, dim=1)

            zslScore = torch.sum(y_out == gt).item()
            zslAcc   = zslScore / n_unseen
            print("ZSL Acc            : %s" % str(zslAcc))
            # -------------------------------------------------------------------------------------------------------- #


            # -------------------------------------------------------------------------------------------------------- #
            # GENERALIZED TRAINING ACCURACY

            x   = torch.from_numpy(testFeatures).float()
            gt  = torch.from_numpy(np.squeeze(np.argmax(genTestOneHotLabels, axis=1)))

            y_out = model(x, allVectors)
            y_out = torch.argmax(y_out, dim=1)

            gzslScoreTR = torch.sum(y_out == gt).item()
            gzslAccTR   = gzslScoreTR / n_test
            print("GZSL TR Acc        : %s" % str(gzslAccTR))
            # -------------------------------------------------------------------------------------------------------- #


            # -------------------------------------------------------------------------------------------------------- #
            # GENERALIZED TEST ACCURACY

            x   = torch.from_numpy(unseenFeatures).float()
            gt  = torch.from_numpy(np.squeeze(np.argmax(genUnseenOneHotLabels, axis=1)))

            y_out = model(x, allVectors)
            y_out = torch.argmax(y_out, dim=1)

            gzslScoreTE = torch.sum(y_out == gt).item()
            gzslAccTE   = gzslScoreTE / n_unseen
            print("GZSL TE Acc        : %s" % str(gzslAccTE))
            # -------------------------------------------------------------------------------------------------------- #

            gzslAcc = (2 * gzslAccTR * gzslAccTE) / (gzslAccTR + gzslAccTE)
            print("GZSL Acc           : %s" % str(gzslAcc))
            print("##" * 25)

    return

if __name__ == '__main__':
    main()
