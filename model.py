# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# model.py
#
# Zero-shot model and zero-shot evaluation method
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


class Network(torch.nn.Module):
    """ Zero-Shot model """

    def __init__(self, feature_dim, vector_dim):

        super(Network, self).__init__()
        self.wHidden1   = torch.nn.Linear(feature_dim, vector_dim)

    def forward(self, imageFeatures, classVectors):

        imageFeatures   = self.wHidden1(imageFeatures)
        out             = torch.nn.Softmax(dim=1)(torch.matmul(imageFeatures, torch.t(classVectors)))

        return out


def evaluate(model, x, y, vec):
    """ Normalized Zero-Shot Evaluation Method """

    classIndices    = np.unique(y.numpy())
    n_class         = len(classIndices)
    t_acc           = 0.
    y_preds         = model(x, vec)

    for index in classIndices:

        sampleIndices   = [i for i, _y in enumerate(y) if _y==index]
        n_sample        = len(sampleIndices)
        y_sample_preds  = torch.argmax(y_preds[sampleIndices], dim=1)
        y_samples       = y[sampleIndices]
        sampleScore     = torch.sum(y_sample_preds == y_samples).item()
        sampleAcc       = sampleScore / n_sample
        t_acc           += sampleAcc

    acc = t_acc / n_class
    return acc