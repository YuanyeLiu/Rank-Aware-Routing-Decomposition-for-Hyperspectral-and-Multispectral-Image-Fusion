import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import json
import time
import hdf5storage
import numpy as np

import torch
from scipy.io import loadmat
from torch import nn
import torch.utils.data as data
import metrics
from FusionDataset import FusionDataProcess
from utils import create_F, fspecial, AverageMeter
from scipy import signal

from RASD_FuNet import RASD_FuNet



def Gaussian_downsample(x, psf, s):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    y = np.zeros((x.shape[0], int(x.shape[1] / s), int(x.shape[2] / s)))
    for i in range(x.shape[0]):
        x1 = x[i, :, :]
        x2 = signal.convolve2d(x1, psf, boundary='symm', mode='same')
        y[i, :, :] = x2[0::s, 0::s]
    return y


def reconstruction(net2, HSI_LR, MSI, downsample_factor, training_size, stride):
    index_matrix = torch.zeros((HSI_LR.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    abundance_t = torch.zeros((HSI_LR.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    a = []
    for j in range(0, MSI.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI.shape[2] - training_size)
    b = []
    for j in range(0, MSI.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, :, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            with torch.no_grad():
                out = net2(temp_lrhs,temp_hrms)
                # out, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = net2(temp_lrhs, temp_hrms)   # ssrnet

                assert torch.isnan(out).sum() == 0

                HSI = out.squeeze()
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon


def reconstruction_parallel(net2, HSI_LR, MSI, downsample_factor, training_size, stride, batch_size):
    index_matrix = torch.zeros((HSI_LR.shape[1], MSI.shape[2], MSI.shape[3]))
    abundance_t = torch.zeros((HSI_LR.shape[1], MSI.shape[2], MSI.shape[3]))

    a = []
    for j in range(0, MSI.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI.shape[2] - training_size)

    b = []
    for j in range(0, MSI.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI.shape[3] - training_size)

    # Create a list of small blocks that will be processed in batches
    blocks = []
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, :, int(j / downsample_factor):int((j + training_size) / downsample_factor),
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
            for idx, (out, (temp_lrhs, temp_hrms, j, k)) in enumerate(zip(out_batch, batch)):
                assert torch.isnan(out).sum() == 0

                HSI = out.squeeze().cpu()
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] += HSI
                index_matrix[:, j:j + training_size, k:k + training_size] += 1

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon



def main(args):
  
    model_path = r"/home/yuanye/code/open_code/train_save/CAVE/RASD_FuNet/1/_PSNR_best.pkl"
    save_path = r"/data/yuanyeliu/RASD_FuNet/CAVE/"
    R = create_F()
    PSF = fspecial('gaussian', 8, 3)
    net = RASD_FuNet(3,31,32).cuda()
    checkpoint = torch.load(model_path)  # Load Breakpoint
    net.load_state_dict(checkpoint)

    RMSE = []


    test_data = FusionDataProcess(args.dataname,args.path,args.cache_path, R, args.patchsize, args.stride, args.factor, PSF,"test")
    test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False,num_workers=0, pin_memory=True)

    loss_func = nn.L1Loss(reduction='mean').cuda()

    sam = AverageMeter()
    rmse = AverageMeter()
    psnr = AverageMeter()
    timereocrd=AverageMeter()
    net.eval()
    for cnt,(lrhsi, msi, gt) in enumerate(test_loader):
        if True:
            with torch.no_grad():
                fusion=net(lrhsi.cuda(),msi.cuda())

                # fusion = reconstruction_parallel(net, lrhsi.cuda(), msi.cuda(),args.factor, 64, 32, 32).unsqueeze(0)
                # fusion = reconstruction(net, lrhsi.cuda(), msi.cuda(),args.factor, 512, 500).unsqueeze(0)

            fusion=torch.clamp(fusion,0,1).cuda()
            # print(Fuse.shape)
            psnr_current=metrics.calc_psnr(fusion,gt.cuda())
            sam_current=metrics.calc_sam(fusion.squeeze(0),gt.cuda().squeeze(0))
            psnr.update(psnr_current)
            sam.update(sam_current)

            faker_hyper = np.transpose(fusion.detach().cpu().numpy().squeeze(0), (1, 2, 0))
            gt = np.transpose(gt.detach().cpu().numpy().squeeze(0), (1, 2, 0))

            print(psnr_current,sam_current)

            test_data_path = os.path.join(save_path + "/"+str(cnt))
            # hdf5storage.savemat(test_data_path, {'fak': faker_hyper}, format='7.3')
            # hdf5storage.savemat(test_data_path, {'rea': gt}, format='7.3')

    print("val  PSNR:", psnr.avg, "  RMSE:", "  SAM:", sam.avg)
    print("mean time:",timereocrd.avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train for Fusion')


    # Add arguments
    # data
    parser.add_argument('--dataname', type=str, default=r'', help='Name of the dataset')
    parser.add_argument('--path', type=str, default=r'', help='Path to the dataset or input files')
    parser.add_argument('--factor', type=int, default=8, help='downsample factor for patch extraction (default: 32)')
    parser.add_argument('--patchsize', type=int, default=64, help='Patch size for training (default: 64)')
    parser.add_argument('--stride', type=int, default=32, help='Stride for patch extraction (default: 32)')

    # model
    parser.add_argument('--modelname', type=str, default=r'ours',help='Name of the dataset')
    
    # train
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=2000, help='Train epoch for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--test__freq', type=int, default=20, help='Test frequency for training')
    parser.add_argument('--clip_grad', type=bool, default=False, help='')
    parser.add_argument('--cache_path', type=str, default=r'', help='')

    # Parse arguments
    args = parser.parse_args()

    config_file=r"/home/yuanye/code/open_code/CAVE.josn"
    with open(config_file, 'r') as f:
        config = json.load(f)
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    main(args)