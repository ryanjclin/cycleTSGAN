import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox  
import pywt

'''
check if fre msg (whole sequence) is white noise
actually, when we train the model, we do sliding window, 
we assume if the whole sequence is white noise, the window is white noise
'''
def white_check(data, wavelet_level):
    white_check_list = []

    for i in range(data.shape[1]):
        coeffs = pywt.wavedec(data[:,i], 'db1', level = wavelet_level, mode='sym')
        for j in range(len(coeffs)):
            white_noise_test = acorr_ljungbox(coeffs[j], return_df = True)
            if white_noise_test['lb_pvalue'].values[0] >= 0.05: # it's white noise
                # white_check_list.append(1)
                white_check_list.append(0)
            else: 
                white_check_list.append(0)

    return np.array(white_check_list)

'''
divide non-white noise and white noise
'''
def var_divide_train_keep(data, white_check_list, source_encoding):
    
    data = data.transpose([0, 2, 1])  # [sample_size, seq_len, var_num]    

    data_to_train = []
    data_to_keep = []
    filter_source_encoding = []
    
    for i in range(len(white_check_list)):
        if white_check_list[i] == 0:
            data_to_train.append(data[:,:,i])
            filter_source_encoding.append(source_encoding[i])
        else: 
            data_to_keep.append(data[:,:,i])
            
    data_to_train = np.array(data_to_train).transpose([1, 0, 2]) # [sample_size, var_num, seq_len]
    data_to_keep = np.array(data_to_keep)#.transpose([1, 0, 2])[-1]
    filter_source_encoding = np.array(filter_source_encoding)

    return data_to_train, data_to_keep, filter_source_encoding



def combine_whiteNoise_with_nonWhiteNoise(nonWhiteNoise, whiteNoise, white_noise_record_fault):
    nonWhiteNoise_count = 0
    whiteNoise_count = 0

    '''nonWhiteNoise: [var_num, seq_len] & whiteNoise: [var_num, seq_len] '''

    nonWhiteNoise = nonWhiteNoise.transpose([1, 0])
    whiteNoise = whiteNoise.transpose([1, 0])

    '''nonWhiteNoise: [seq_len, var_num] & whiteNoise: [seq_len, var_num]'''


    complete_data = []
    for i in white_noise_record_fault:
        if i == 0:
            complete_data.append(nonWhiteNoise[:, nonWhiteNoise_count])
            nonWhiteNoise_count = nonWhiteNoise_count + 1
        else: 
            complete_data.append(whiteNoise[:, whiteNoise_count])
            whiteNoise_count = whiteNoise_count + 1   

    complete_data = np.array(complete_data)

    return complete_data