import torch
import pandas as pd
import numpy as np

from model.generator import Generator
from utils.window_slide_wavelet import add_window_wavelet, reverse_wavelet
from utils.white_noise_check import var_divide_train_keep, combine_whiteNoise_with_nonWhiteNoise
from utils.simple_operation import normalize_fre, normalize_fre_reverse

def inferencing(config, tep_normal, device, fault_id, preprocess_result):

    # load normal data for fault transformation
    time_noral = tep_normal[400: 400 + (config['window_size'] + config['batch_size']) ]

    # window slidind and wavelet
    fre_normal, fre_msg_length_record, source_encoding = add_window_wavelet(time_noral, config['window_size'], config['wavelet_level'])

    # normalization
    # fre_normal_norm, mean, std = normalize_fre(fre_normal)
    fre_normal_norm, _, _ = normalize_fre(fre_normal)

    # load trained model
    gen_NormalToFault = Generator(config, source_encoding).to(device) # normal to fault
    gen_NormalToFault.load_state_dict(torch.load(config['checkpoint'] + f"/gen_NormalToFault_32000.bin"))  # use the checkpoint you want
    gen_NormalToFault.eval()

    # generation
    fre_normal_norm = torch.Tensor(fre_normal_norm).to(device)
    gen_fre_fault = gen_NormalToFault(fre_normal_norm)

    # reverse normalization
    # gen_fre_fault = normalize_fre_reverse(gen_fre_fault.data.cpu().numpy(), mean, std)
    gen_fre_fault = normalize_fre_reverse(gen_fre_fault.data.cpu().numpy(), preprocess_result['data_mean'], preprocess_result['data_std'])
    gen_fre_fault = gen_fre_fault[0]

    # wavelet composition: reverse wavelet decomposition, converting fre domain back to time domain
    fre_msg_num = config['wavelet_level'] + 1 # num of fre pieces is wavelet_level + 1
    gen_time_series = reverse_wavelet(gen_fre_fault, fre_msg_num, fre_msg_length_record)

    # save gen_data as csv
    df = pd.DataFrame(gen_time_series)
    print(f'gen_data: {df.shape}')
    df.to_csv(config['generated_data'] + f'/gen_data_{fault_id}.csv')









    # normalization
    # fre_normal_norm, _, _ = normalize_fre(fre_normal)

    # reverse normalization
    # gen_fre_fault = normalize_fre_reverse(gen_fre_fault.data.cpu().numpy(), preprocess_result['data_mean'], preprocess_result['data_std'])
