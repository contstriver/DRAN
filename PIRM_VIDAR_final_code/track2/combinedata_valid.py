from __future__ import division
import numpy as np
from scipy.ndimage import imread
import h5py
import os

from tools.utils import save_matv73


track1_data_path = './track1_valid_data'
track2_data_path = './track2_valid_data'
var_name = 'data'
result_path = './track2_final_valid_data'
if not os.path.exists(result_path):
    os.makedirs(result_path)


n_crop= 12

for mat_name in sorted(os.listdir(track1_data_path)):
    track1_path_name = os.path.join(track1_data_path, mat_name)
    f = h5py.File(track1_path_name,'r')
    track1_data = f.get(var_name)
    track1_data = np.array(track1_data) 
    track1_data = np.transpose(track1_data,[2,1,0])

    track2_path_name = os.path.join(track2_data_path, mat_name)
    f = h5py.File(track2_path_name,'r')
    track2_data = f.get(var_name)
    track2_data = np.array(track2_data) 
    track2_data = np.transpose(track2_data,[2,1,0])

    comb_data = np.zeros(track1_data.shape) 
    comb_data = track1_data 
    comb_data[n_crop:track1_data.shape[0]-n_crop,n_crop:track1_data.shape[1]-n_crop,:] = track2_data 


    mat_dir= os.path.join(result_path, mat_name)
    save_matv73(mat_dir, var_name, comb_data)
   
    print(mat_name)
    print(track1_data.shape)
    print(track2_data.shape)
    print(comb_data.shape)
