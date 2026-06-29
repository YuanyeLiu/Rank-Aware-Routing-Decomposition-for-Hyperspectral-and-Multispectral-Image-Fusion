import argparse
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import torch
import metrics
from utils import AverageMeter
import numpy as np
from scipy.io import loadmat
import hdf5storage as h5
from torch import nn
from tqdm import tqdm
import time
import torch.utils.data as data
import math
from scipy import signal
from torch.utils.data import Dataset

from RASD_FuNet import RASD_FuNet


def reconstruction_fg5_parallel(net2, HSI_LR, MSI_HR, downsample_factor, training_size, stride, batch_size):
    index_matrix = torch.zeros((HSI_LR.shape[1], MSI_HR.shape[2], MSI_HR.shape[3]),dtype=torch.float16)
    abundance_t = torch.zeros((HSI_LR.shape[1], MSI_HR.shape[2], MSI_HR.shape[3]),dtype=torch.float16)

    a = []
    for j in range(0, MSI_HR.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI_HR.shape[2] - training_size)

    b = []
    for j in range(0, MSI_HR.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI_HR.shape[3] - training_size)

    # Create a list of small blocks that will be processed in batches
    blocks = []
    for j in a:
        for k in b:
            temp_hrms = MSI_HR[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, :,
                               int(j / downsample_factor):int((j + training_size) / downsample_factor),
                               int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            blocks.append((temp_lrhs, temp_hrms, j, k))

    # Process the blocks in batches
    for i in range(0, len(blocks), batch_size):
        batch = blocks[i:i + batch_size]
        temp_lrhs_batch = torch.stack([block[0] for block in batch]).cuda()
        temp_hrms_batch = torch.stack([block[1] for block in batch]).cuda()

        with torch.no_grad():
            # Forward pass on the batch of small blocks
            out_batch = net2(temp_lrhs_batch.squeeze(1), temp_hrms_batch.squeeze(1))

            # Process each output and accumulate in index_matrix and abundance_t
            for out, (temp_lrhs, temp_hrms, j, k) in zip(out_batch, batch):
                assert torch.isnan(out).sum() == 0
                HSI = out.squeeze().cpu()
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j :j + training_size , k :k + training_size ] += HSI
                index_matrix[:, j :j + training_size , k :k + training_size ] += 1

    HSI_recon = abundance_t / index_matrix
    # assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon



def main(args):
    HSI= np.load(args.hsi_path)   
    MSI= np.load(args.msi_path)     
    HSI = np.transpose(HSI, (2, 0, 1))
    MSI = np.transpose(MSI, (2, 0, 1))
    print("Training has finished loading")


    cnn= RASD_FuNet(4,150,128).cuda()

    model_path = r"/home/yuanye/code/open_code/train_save/GF5/RASD_FuNet/1/_PSNR_best.pkl"
    save_path = r"/data/yuanyeliu/RASD_FuNet/GF5/"
    cnn.load_state_dict(torch.load(model_path))

    cnn.eval()
   
    with torch.no_grad():
        test_HRMSI =torch.unsqueeze(torch.Tensor(MSI[:, :1100, :1144]),0)
        test_LRHSI=torch.unsqueeze(torch.Tensor(HSI[:, :550, :572]),0)

        fusion=reconstruction_fg5_parallel(cnn, test_LRHSI.cuda(), test_HRMSI.cuda(),args.factor, 128, 96,32)
        # fusion=cnn(test_LRHSI.cuda(), test_HRMSI.cuda())
        fusion = torch.clamp(fusion, 0, 1)

        # print(Fuse.shape)
        # fusion=torch.round(fusion*255)/255.0

        test_HRMSI = test_HRMSI.squeeze(0).cpu().float().permute(1, 2, 0).numpy()
        test_LRHSI = test_LRHSI.squeeze(0).cpu().float().permute(1, 2, 0).numpy()
        fusion = fusion.squeeze(0).cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()

        test_data_path = os.path.join(save_path + "/"+"gf5_1100_1144")
        # h5.savemat(test_data_path, {'Z': fusion}, format='7.3')
        # h5.savemat(test_data_path, {'pan': test_HRMSI}, format='7.3')
        # h5.savemat(test_data_path, {'lr': test_LRHSI}, format='7.3')
        print("Done")






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train for Fusion')


    # Add arguments
    # data
    parser.add_argument('--dataname', type=str, default=r'', help='Name of the dataset')
    parser.add_argument('--path', type=str, default=r'', help='Path to the dataset or input files')
    parser.add_argument('--factor', type=int, default=8, help='downsample factor for patch extraction (default: 32)')
    parser.add_argument('--patchsize', type=int, default=64, help='Patch size for training (default: 64)')
    parser.add_argument('--stride', type=int, default=32, help='Stride for patch extraction (default: 32)')

    parser.add_argument('--hsi_path', type=str, default=r'', help='Path to the dataset or input files')
    parser.add_argument('--msi_path', type=str, default=r'', help='Path to the dataset or input files')
    parser.add_argument('--R_path', type=str, default=r'', help='Path to the dataset or input files')
    parser.add_argument('--C_path', type=str, default=r'', help='Path to the dataset or input files')

    # model
    parser.add_argument('--modelname', type=str, default=r'ours',help='Name of the dataset')
    
    # train
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=2000, help='Train epoch for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--test__freq', type=int, default=20, help='Test frequency for training')
    parser.add_argument('--clip_grad', type=bool, default=False, help='')
 
    # Parse arguments
    args = parser.parse_args()

    config_file=r"/home/yuanye/code/open_code/GF5.josn"
    with open(config_file, 'r') as f:
        config = json.load(f)
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    main(args)
