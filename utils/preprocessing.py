# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data as Data
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os, sys, time
# import pywt
# from statsmodels.stats.diagnostic import acorr_ljungbox  
# from fastdtw import fastdtw
# from collections import defaultdict 

from .simple_operation import normalize_fre
from .white_noise_check import white_check, var_divide_train_keep
from .window_slide_wavelet import add_window_wavelet

'''
1. do wavelet
2. check if a variable is white noise, if it is, keep and do not feed it into model.
3. normalization
'''
def data_preprocessing(data_normal, data_fault, config):

    '''
    check if fre msg (whole sequence) is white noise
    actually, when we train the model, we do sliding window, instead of the whole sequence
    we assume if the whole sequence is white noise, the window is white noise
    record the white noise variable (fre msg) 
    '''
    white_noise_record_fault = white_check(data_fault, config['wavelet_level'])

    # add sliding window and do wavelet
    fre_faulty, _ = add_window_wavelet(data_fault, config['window_size'], config['wavelet_level']) # [sample_size, var_num, seq_len]
    fre_normal, _ = add_window_wavelet(data_normal, config['window_size'], config['wavelet_level']) # [sample_size, var_num, seq_len]

    '''
    divide white noise fre msg and non-white noise msg
    only use non-white noise msg to train model, because white noise fre msg does not contain useful msg
    '''
    fre_faulty, fre_faulty_white_noise = var_divide_train_keep(fre_faulty, white_noise_record_fault)
    fre_normal, _ = var_divide_train_keep(fre_normal, white_noise_record_fault)

    # normalize fre msg
    fre_faulty_norm, data_mean, data_std = normalize_fre(fre_faulty)
    fre_normal_norm, _, _ = normalize_fre(fre_normal)

    preprocess_result = {
        'fre_faulty_norm': fre_faulty_norm,
        'fre_normal_norm': fre_normal_norm,
        'white_noise_record_fault': white_noise_record_fault,
        'data_mean': data_mean,
        'data_std': data_std,
        'fre_faulty_white_noise': fre_faulty_white_noise,
    }

    return preprocess_result
