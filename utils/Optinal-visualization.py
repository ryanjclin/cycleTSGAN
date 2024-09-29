import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw

'''
visualization: help you to understande the tep time series
'''

tep00 = np.genfromtxt("tennessee-eastman-profBraatz-master/d00_te.dat")[:400]
tep13 = np.genfromtxt("tennessee-eastman-profBraatz-master/d01_te.dat")[:400]

# tep00_mean = np.mean(tep00,0)   #(72,)
# tep00_std = np.std(tep00,0,ddof=1) 
# tep00 = (tep00 - tep00_mean)/tep00_std

# tep13_mean = np.mean(tep13,0)   #(72,)
# tep13_std = np.std(tep13,0,ddof=1) 
# tep13 = (tep13 - tep13_mean)/tep13_std

dist_set = []
for i in range(tep13.shape[1]):
    distance, path = fastdtw(tep00[:,i], tep13[:,i])
    dist_set.append(distance)
dist_set = np.array(dist_set)
print(dist_set.shape)

for i in range(dist_set.shape[0]):
    plt.rcParams['figure.figsize'] = [8, 4]
    plt.figure(i) 
    plt.plot(tep00[:,int(np.argsort(dist_set)[i])],'r')
    plt.plot(tep13[:,int(np.argsort(dist_set)[i])])