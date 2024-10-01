from .simple_operation import normalize_fre
from .white_noise_check import white_check, var_divide_train_keep
from .window_slide_wavelet import add_window_wavelet

'''
1. do wavelet
2. normalization
'''
def data_preprocessing(data_normal, data_fault, config):

    # add sliding window and do wavelet
    fre_faulty, _, source_encoding = add_window_wavelet(data_fault, config['window_size'], config['wavelet_level']) # [sample_size, var_num, seq_len]
    fre_normal, _, _ = add_window_wavelet(data_normal, config['window_size'], config['wavelet_level']) # [sample_size, var_num, seq_len]

    # normalize fre msg
    fre_faulty_norm, data_mean, data_std = normalize_fre(fre_faulty)
    fre_normal_norm, _, _ = normalize_fre(fre_normal)


    '''
    fre_faulty_norm: [sample_size, var_num, seq_len]
    fre_normal_norm: [sample_size, var_num, seq_len]
    data_mean: [var_num, 1]
    data_std: [var_num, 1]
    '''

    preprocess_result = {
        'fre_faulty_norm': fre_faulty_norm,
        'fre_normal_norm': fre_normal_norm,
        'data_mean': data_mean,
        'data_std': data_std,
        'source_encoding': source_encoding,
    }

    return preprocess_result
