# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# tools.py
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# April, 2019
# --------------------------------------------------
import random
random.seed(123)

import numpy as np
np.random.seed(123)

import scipy.io as sio


def load_data(data, dataName):
    """ Load data from .mat files """

    dataContent = sio.loadmat(data)
    dataContent = dataContent[dataName]

    return dataContent

def map_labels(labels, n_class, indices_to_map):
    """ Map given sample label to corresponding label """

    n_label         = len(labels)
    labels_one_hot  = np.zeros(shape=(n_label, n_class))

    for i, label in enumerate(labels):
        labels_one_hot[i][label[0]] = 1.

    mappedLabels =  np.argmax(labels_one_hot[:, indices_to_map], axis=1)

    return mappedLabels