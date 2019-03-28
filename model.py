# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# model.py
#
# Attribute Label Embedding (ALE) Method
# Written by cetinsamet
# March, 2019
# --------------------------------------------------

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)

from torch import nn
import torch.nn.functional as F



class Network(nn.Module):

    def __init__(self, feature_dim, vector_dim, n_hidden):

        super(Network, self).__init__()
        self.wHidden1   = nn.Linear(feature_dim, n_hidden)
        self.wHidden2   = nn.Linear(vector_dim, n_hidden)

    def forward(self, imageFeatures, classVectors):

        imageFeatures   = self.wHidden1(imageFeatures)
        classVectors    = F.relu(self.wHidden2(classVectors))
        out             = nn.Softmax(dim=1)(torch.matmul(imageFeatures, torch.t(classVectors)))

        return out