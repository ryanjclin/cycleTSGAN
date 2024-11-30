import torch
import pandas as pd
import numpy as np

from model.generator import Generator
from utils.window_slide_wavelet import add_window_wavelet, reverse_wavelet
from utils.white_noise_check import var_divide_train_keep, combine_whiteNoise_with_nonWhiteNoise
from utils.simple_operation import normalize_fre, normalize_fre_reverse, normalize_fre_infer

def inferencing(config, tep_normal, tep_fault, device, fault_id, preprocess_result):

    # load normal data for fault transformation
    time_normal = tep_normal[config['start_time_id']: config['end_time_id'] + config['window_size']]  # (sample_size, 52)
    time_fault = tep_fault[config['start_time_id']: config['end_time_id'] + config['window_size']]  # (sample_size, 52)

    print(f"time_normal: {time_normal.shape}")
    print(f"time_fault: {time_normal.shape}")

    # window slidind and wavelet (this does sliding window + wavelet decomposition)
    fre_normal, fre_msg_length_record, source_encoding = add_window_wavelet(time_normal, config['window_size'], config['wavelet_level'])

    # do wavelet decomposition only
    time_normal_windows = np.vstack([np.expand_dims(time_normal[i:i + config['window_size']], axis=0) for i in range(time_normal.shape[0] - config['window_size'])])
    print(f"time_normal_windows: {time_normal_windows.shape}")
    np.save(f'eval_data/real_normal.npy', time_normal_windows)

    time_fault_windows = np.vstack([np.expand_dims(time_fault[i:i + config['window_size']], axis=0) for i in range(time_fault.shape[0] - config['window_size'])])
    print(f"time_fault_windows: {time_fault_windows.shape}")
    np.save(f'eval_data/real_fault_{fault_id}.npy', time_fault_windows)

    # normalization
    # fre_normal_norm, _, _ = normalize_fre(fre_normal)
    fre_normal_norm = normalize_fre_infer(fre_normal, preprocess_result['noraml_mean'], preprocess_result['normal_std'])

    # load trained model
    gen_NormalToFault = Generator(config, source_encoding).to(device) # normal to fault
    gen_NormalToFault.load_state_dict(torch.load(config['checkpoint'] + f"/gen_NormalToFault_{fault_id}.bin", map_location=torch.device('cuda:0')))  # use the checkpoint you want
    gen_NormalToFault.eval()

    # generation
    fre_normal_norm = torch.Tensor(fre_normal_norm).to(device)
    gen_fre_fault = gen_NormalToFault(fre_normal_norm)

    print(f"gen_fre_fault: {gen_fre_fault.shape}")

    # reverse normalization
    gen_fre_fault = normalize_fre_reverse(gen_fre_fault.data.cpu().numpy(), preprocess_result['faulty_mean'], preprocess_result['faulty_std'])
    print(f"gen_fre_fault: {gen_fre_fault.shape}")

    # TODO: wavelet decomposition for the entire batch
    gen_fre_fault_time_series = None
    fre_msg_num = config['wavelet_level'] + 1 # num of fre pieces is wavelet_level + 1
    for i in range(gen_fre_fault.shape[0]):
        time_series = reverse_wavelet(gen_fre_fault[i], fre_msg_num, fre_msg_length_record)
        if gen_fre_fault_time_series is None:
            gen_fre_fault_time_series = time_series[np.newaxis, :]
        else:
            gen_fre_fault_time_series = np.vstack((gen_fre_fault_time_series, time_series[np.newaxis, :]))
    
    print(gen_fre_fault_time_series.shape)
    np.save(f'eval_data/gen_fault_{fault_id}.npy', gen_fre_fault_time_series)

    # gen_fre_fault = gen_fre_fault[0]
    # print(f"gen_fre_fault: {gen_fre_fault.shape}")

    # # wavelet composition: reverse wavelet decomposition, converting fre domain back to time domain
    # fre_msg_num = config['wavelet_level'] + 1 # num of fre pieces is wavelet_level + 1
    # gen_time_series = reverse_wavelet(gen_fre_fault, fre_msg_num, fre_msg_length_record)

    # # save gen_data as csv
    # df = pd.DataFrame(gen_time_series)
    # print(f'gen_data: {df.shape}')
    # df.to_csv(config['generated_data'] + f'/gen_data_{fault_id}.csv')









    # normalization
    # fre_normal_norm, _, _ = normalize_fre(fre_normal)

    # reverse normalization
    # gen_fre_fault = normalize_fre_reverse(gen_fre_fault.data.cpu().numpy(), preprocess_result['data_mean'], preprocess_result['data_std'])
