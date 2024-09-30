import torch
import pandas as pd

from model.generator import Generator
from utils.window_slide_wavelet import add_window_wavelet, reverse_wavelet
from utils.white_noise_check import var_divide_train_keep, combine_whiteNoise_with_nonWhiteNoise
from utils.simple_operation import normalize_fre, normalize_fre_reverse

def inferencing(config, tep_normal, device, preprocess_result, fault_id):

    # load trained model
    gen_NormalToFault = Generator(config['batch_size'], config['var_num'], config['seq_len'], preprocess_result['filtered_source_encoding']).to(device) # normal to fault
    gen_NormalToFault.load_state_dict(torch.load(config['checkpoint'] + f"/gen_NormalToFault_10000_L1.bin"))  # use the checkpoint you want
    gen_NormalToFault.eval()

    # load normal data for fault transformation
    time_noral = tep_normal[400: 400 + (config['window_size'] + config['batch_size']) ]

    # window slidind and wavelet
    fre_normal, fre_msg_length_record, source_encoding = add_window_wavelet(time_noral, config['window_size'], config['wavelet_level'])

    # remove white noise
    fre_normal, mean, std = var_divide_train_keep(fre_normal, preprocess_result['white_noise_record_fault'], source_encoding)
    # fre_normal, _, _ = var_divide_train_keep(fre_normal, preprocess_result['white_noise_record_fault'], source_encoding)

    # normalization
    # fre_normal_norm, mean, std = normalize_fre(fre_normal)
    fre_normal_norm, _, _ = normalize_fre(fre_normal)

    # generation
    gen_fre_fault = gen_NormalToFault(torch.Tensor(fre_normal_norm).to(device)).to(device)

    # reverse normalization
    gen_fre_fault = normalize_fre_reverse(gen_fre_fault.data.cpu().numpy(), mean, std)
    # gen_fre_fault = normalize_fre_reverse(gen_fre_fault.data.cpu().numpy(), preprocess_result['data_mean'], preprocess_result['data_std'])

    # only use the last window synthetic time series
    gen_fre_fault = gen_fre_fault[-1]
    ori_fre_fault_white_noise = preprocess_result['fre_faulty_white_noise']

    # combine gen_fre_fault and ori_fre_fault_white_noise
    complete_gen_fre_msg = combine_whiteNoise_with_nonWhiteNoise(gen_fre_fault, ori_fre_fault_white_noise, preprocess_result['white_noise_record_fault'])
    
    # wavelet composition: reverse wavelet decomposition, converting fre domain back to time domain

    fre_msg_num = config['wavelet_level'] + 1 # num of fre pieces is config['wavelet_level']+1
    gen_time_series = reverse_wavelet(complete_gen_fre_msg, fre_msg_num, fre_msg_length_record)

    # save gen_data as csv
    df = pd.DataFrame(gen_time_series)
    print(f'gen_data: {df.shape}')
    df.to_csv(config['generated_data'] + f'/gen_data_{fault_id}.csv')



