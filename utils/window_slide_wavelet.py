import numpy as np
import pywt

'''
we do not discard low fre msg, but use all fre msg
later, we do white noise chech to remove white noise msg no matter it is high or low fre
'''

def add_window_wavelet(time_series, time_step, wavelet_level):

    frq_msg_window = []
    source_encoding = []
    fre_msg_length_record = []

    for i in range(len(time_series)-time_step): # sliding window
        dat = time_series[i:i+time_step] 

        fre_all_var = []

        for j in range(dat.shape[1]):  # iterate every variable to do wavelet decomposition
            coeffs = pywt.wavedec(dat[:,j], 'db1', level = wavelet_level, mode='sym') # a time series converted into multiple fre pieces, each of which contains a certrain fre msg

            longest_coeffs = len(coeffs[-1])
            
            fre_one_var = []
            for k in range(len(coeffs)): # because each fre piece's len is different, we need to make its len uniform.
                
                multiplier = int(longest_coeffs / len(coeffs[k]))

                uniform_high_fre = list(coeffs[k])
                for _ in range(multiplier-1):  # repeat the short one until its len aligns with the longest one
                    uniform_high_fre += list(coeffs[k])

                fre_one_var.append(uniform_high_fre)
                
                if j == 0 and i == 0:
                    fre_msg_length_record.append(len(coeffs[k]))

                if i == 0:
                    source_encoding.append(j)

            #     print(f"len(coeffs[k]): {len(coeffs[k])}")
            # raise

            fre_all_var.append(fre_one_var)

        fre_all_var = np.array(fre_all_var).reshape([-1, longest_coeffs])
        
        frq_msg_window.append(fre_all_var)
        
    frq_msg_window = np.array(frq_msg_window) # [sample_size, var_num, seq_len]
    # frq_msg_window = frq_msg_window.transpose([0, 2, 1])  # [sample_size, seq_len, var_num]    

    return frq_msg_window, fre_msg_length_record, np.array(source_encoding)



# wavelet composition: reverse wavelet decomposition, converting fre domain back to time domain
def reverse_wavelet(data, fre_msg_num, fre_msg_length_record):

    data = data.reshape([int(len(data)//fre_msg_num), fre_msg_num, -1]) #[var_num, fre_msg_num, seq_len]

    wavelet_reverse_data = []
    for i in range(len(data)):
        dat = data[i]

        coeffes_reverse = []
        for j in range(fre_msg_num):
            all_fre_msg = dat[j]
            ori_fre_len = fre_msg_length_record[j]

            window_fre_msg = [all_fre_msg[i:i+ori_fre_len] for i in range(0, len(all_fre_msg), ori_fre_len)]

            average_fre_msg = 0
            for fre_msg in window_fre_msg:
                average_fre_msg += fre_msg
            average_fre_msg = average_fre_msg / len(window_fre_msg)

            # print(f"---------")
            # print(f"all_fre_msg: {all_fre_msg.shape}")
            # print(f"ori_fre_len: {ori_fre_len}")
            # print(f" len(window_fre_msg: { len(window_fre_msg)}")
            # print(f"---------")

            coeffes_reverse.append(average_fre_msg)
        data_reverse = pywt.waverec(coeffes_reverse, 'db1', mode='sym')
        wavelet_reverse_data.append(data_reverse)
    wavelet_reverse_data = np.array(wavelet_reverse_data).T

    return wavelet_reverse_data