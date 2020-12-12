from __future__ import division
import torch
import torch.nn as nn

import os
import numpy as np
from scipy.ndimage import imread
import h5py

from network.net_track1 import make_model
from tools.utils import save_matv73, AverageMeter

class args:
    def __init__(self):
        super(CALayer, self).__init__()
        args.n_resgroups 
        args.n_resblocks
        args.n_feats 
        args.n_reduction

def cal_mrae(target,img_res):       
    diff = target - img_res
    abs_diff = np.abs(diff)
    relative_abs_diff = np.divide(abs_diff,target + 1) 
    MRAEs = np.mean(relative_abs_diff)
#     ax = sns.heatmap(MRAEs[:,:,1],vmin=0, vmax=1)
    return MRAEs

def postprocess(img_res):
    img_res = torch.clamp(img_res*65535,max=65535,min=0)
    img_res = torch.round(img_res)
    img_res = np.squeeze(np.transpose(torch.Tensor.cpu(img_res).detach().numpy(),[3,2,1,0]),axis=3)
    return img_res

def self_ensemble(model,input_data,target):
    input_data1 = input_data
    input_data2 = np.flip(input_data,2)
    
    input_data3 = np.rot90(input_data1, k=1, axes=(2, 1))  
    input_data4 = np.rot90(input_data1, k=2, axes=(2, 1))
    input_data5 = np.rot90(input_data1, k=3, axes=(2, 1))
    
    input_data6 = np.rot90(input_data2, k=1, axes=(2, 1))
    input_data7 = np.rot90(input_data2, k=2, axes=(2, 1))
    input_data8 = np.rot90(input_data2, k=3, axes=(2, 1))
    
    input_data1 = np.expand_dims(input_data1, axis=0).copy()
    input_data2 = np.expand_dims(input_data2, axis=0).copy()
    input_data3 = np.expand_dims(input_data3, axis=0).copy()
    input_data4 = np.expand_dims(input_data4, axis=0).copy()
    input_data5 = np.expand_dims(input_data5, axis=0).copy()
    input_data6 = np.expand_dims(input_data6, axis=0).copy()
    input_data7 = np.expand_dims(input_data7, axis=0).copy()
    input_data8 = np.expand_dims(input_data8, axis=0).copy()
   
    input_data1 = torch.from_numpy(input_data1).float().cuda()
    input_data2 = torch.from_numpy(input_data2).float().cuda()
    input_data3 = torch.from_numpy(input_data3).float().cuda()
    input_data4 = torch.from_numpy(input_data4).float().cuda()
    input_data5 = torch.from_numpy(input_data5).float().cuda()
    input_data6 = torch.from_numpy(input_data6).float().cuda()
    input_data7 = torch.from_numpy(input_data7).float().cuda()
    input_data8 = torch.from_numpy(input_data8).float().cuda()

    
    img_res1 = model(input_data1)
    img_res1 = postprocess(img_res1)

    img_res2 = model(input_data2)
    img_res2 = postprocess(img_res2)
    img_res2 = np.flip(img_res2,0)
    
    img_res3 = model(input_data3)
    img_res3 = postprocess(img_res3)
    img_res3 = np.rot90(img_res3, k=3, axes=(0, 1))
    
    img_res4 = model(input_data4)
    img_res4 = postprocess(img_res4)
    img_res4 = np.rot90(img_res4, k=2, axes=(0, 1))
    
    img_res5 = model(input_data5)
    img_res5 = postprocess(img_res5)
    img_res5 = np.rot90(img_res5, k=1, axes=(0, 1))
    
    img_res6 = model(input_data6)
    img_res6 = postprocess(img_res6)
    img_res6 = np.flip(img_res6,0)
    img_res6 = np.rot90(img_res6, k=1, axes=(0, 1))                       
    
    img_res7 = model(input_data7)
    img_res7 = postprocess(img_res7)
    img_res7 = np.flip(img_res7,0)
    img_res7 = np.rot90(img_res7, k=2, axes=(0, 1)) 
    
    img_res8 = model(input_data8)
    img_res8 = postprocess(img_res8)
    img_res8 = np.flip(img_res8,0)
    img_res8 = np.rot90(img_res8, k=3, axes=(0, 1)) 
    return np.round((img_res1+img_res2+img_res3+img_res4+img_res5+img_res6+img_res7+img_res8)/8)

def main():
    args.n_resgroups = 5
    args.n_resblocks = 3
    args.n_feats = 64
    args.n_reduction = 16

    data_path = './data/valid/lr3'
    gt_path = './data/valid/hr'
    result_path = './track1_valid_data/'
    var_name = 'data'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    model_path = './model/track1_model.pkl'
    save_point = torch.load(model_path)
    model_param = save_point['state_dict']
    model = make_model(args)
    model.load_state_dict(model_param)

    model = model.cuda()
    model.eval()

    mrae = AverageMeter()

    for mat_name in sorted(os.listdir(data_path)):
        mat_path_name = os.path.join(data_path, mat_name)
        f = h5py.File(mat_path_name,'r')
        input_data = f.get(var_name)
        input_data = np.array(input_data) 
        
        mat_name = mat_name[:-8] + '_tr1.mat'
        mat_path_name = os.path.join(gt_path, mat_name)
        f = h5py.File(mat_path_name,'r')
        target = f.get(var_name)
        target = np.array(target) 
        target = np.transpose(target,[2,1,0])
            
        input_data = input_data/65535

        img_res = self_ensemble(model,input_data,target)
        MRAEs = cal_mrae(target,img_res)

        mat_name = mat_name[:-8] + '_tr1.mat'
        mat_dir= os.path.join(result_path, mat_name)
        save_matv73(mat_dir, var_name, img_res)

        mrae.update(MRAEs)
        print(mat_name)
        print(img_res.shape)
        print(MRAEs)
    print(mrae.avg)


if __name__ == '__main__':
    main()
