import numpy as np
import matplotlib.pyplot as plt

'''frequency domian normalization'''
def normalize_fre(data): 
    # data: [sample_size, var_num, seq_len]
    tep_high_fre_norm = []

    last_window = data[-1,:,:] #[var_num, seq_len]

    mean = np.mean(last_window, axis=1) # do normalization on seq_len dimension
    mean = np.expand_dims(mean, axis=1)
    std = np.std(last_window, axis=1)
    std = np.expand_dims(std, axis=1)

    for i in range(len(data)):
        normalized_tensor = (data[i,:,:] - mean) / std
        tep_high_fre_norm.append(normalized_tensor)
        
    return np.array(tep_high_fre_norm), mean, std


''' reverse frequency domian normalization '''
def normalize_fre_reverse(batch_data, mean, std):  
    
    data_reverse = []
    for i in range(len(batch_data)):
        data = batch_data[i,:,:]
        data = data * std + mean
        data_reverse.append(data)
        
    return np.array(data_reverse)

# def normalize_fre_reverse(normalized_tensor, mean, std):
#     # 恢復原始數值
#     original_tensor = normalized_tensor * std + mean
#     return original_tensor