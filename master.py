# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# master.py
#
# Main program that operates validation and test phases of
# attribute label embbedding (ALE) algorithm on APY dataset
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# April, 2019
# --------------------------------------------------
from easydict import EasyDict as edict
from config import MAIN_DATAPATH, VAL_DATAPATH, TEST_DATAPATH, OBJPATH
from decimal import Decimal
import numpy as np
import subprocess
import argparse
import pickle


def prepareData():

    __C = edict()

    __C.LR                  = 1e-2
    __C.BATCH_SIZE          = 64
    __C.N_EPOCH             = 200
    __C.INFO_EPOCH          = 1

    __C.ALL_CLASS_VEC           = MAIN_DATAPATH + 'all_class_vec.mat'

    __C.TRAIN_FEATURES          = VAL_DATAPATH + 'train_features.mat'
    __C.TRAIN_LABELS            = VAL_DATAPATH + 'train_labels.mat'

    __C.VAL_SEEN_FEATURES       = VAL_DATAPATH + 'val_seen_features.mat'
    __C.VAL_SEEN_LABELS         = VAL_DATAPATH + 'val_seen_labels.mat'

    __C.VAL_UNSEEN_FEATURES     = VAL_DATAPATH + 'val_unseen_features.mat'
    __C.VAL_UNSEEN_LABELS       = VAL_DATAPATH + 'val_unseen_labels.mat'

    __C.TRAINVAL_FEATURES       = TEST_DATAPATH + 'trainval_features.mat'
    __C.TRAINVAL_LABELS         = TEST_DATAPATH + 'trainval_labels.mat'

    __C.TEST_SEEN_FEATURES      = TEST_DATAPATH + 'test_seen_features.mat'
    __C.TEST_SEEN_LABELS        = TEST_DATAPATH + 'test_seen_labels.mat'

    __C.TEST_UNSEEN_FEATURES    = TEST_DATAPATH + 'test_unseen_features.mat'
    __C.TEST_UNSEEN_LABELS      = TEST_DATAPATH + 'test_unseen_labels.mat'

    return __C


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", \
                        choices=["validation", "test"], \
                        default="validation", \
                        help="select training phase (validation or test)")
    args = parser.parse_args()


    #'''
    __C = prepareData()  # <---- Load data

    with open(OBJPATH, 'wb') as outfile:
        pickle.dump(__C, outfile, pickle.HIGHEST_PROTOCOL)

    if args.mode == 'validation':  # <-- Perform training on VALIDATION SET
        subprocess.call('python3 validation.py', shell=True)

    elif args.mode == 'test':  # <-- Perform training on TEST SET
        subprocess.call('python3 test.py', shell=True)
    else:
        pass
    #'''