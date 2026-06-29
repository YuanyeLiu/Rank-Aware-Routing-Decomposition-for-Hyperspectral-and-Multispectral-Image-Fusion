
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import sys
# from scipy.io import loadmat
from thop import profile, clever_format
import metrics
from FusionDataset import FusionDataProcess, Gaussian_downsample
from torch import nn
from tqdm import tqdm
import time
# import pandas as pd
import argparse
from utils import AverageMeter, create_F, fspecial
import torch
import numpy as np
import torch.utils.data as data
import random

from RASD_FuNet import RASD_FuNet

# Pin All Seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Configuring CuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialization function for data loader workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  
        os.makedirs(path) 
        print("The training folder is:{}".format(path))
    else:
        print('Already exists{}'.format(path))

loss_func=nn.L1Loss()

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
                # Remove the 1 dimensional dimension
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon



def reconstruction_parallel(net2, HSI_LR, MSI, downsample_factor, training_size, stride, batch_size):
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

                HSI = out.squeeze()
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] += HSI
                index_matrix[:, j:j + training_size, k:k + training_size] += 1

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon


def main(args):
    # Path Parameters
    # root=os.getcwd()
    root=os.path.dirname(os.path.abspath(__file__))
    save_path=os.path.join(root,'train_save',args.dataname)
    save_path_model=os.path.join(save_path,args.modelname)
    mkdir(save_path_model)


    set = {'path': args.path,'patchsize': args.patchsize,'dataname': args.dataname,'stride': args.stride,'lr': args.lr,}

    current_list=os.listdir(save_path_model)
    int_list = [int(folder) for folder in current_list if folder.isdigit()]
    train_value = max(int_list, default=0) + 1 
    save_path_model_num=os.path.join(save_path_model,str(train_value))
    mkdir(save_path_model_num)

    train_record_json = os.path.join(save_path_model_num, 'train_record.json')
    test_record_json = os.path.join(save_path_model_num, 'test_record.json')


    record = []
    if os.path.exists(train_record_json):
        with open(train_record_json, "r") as f:
            record = json.load(f)
    record.append(set)
    # Use the `separators` parameter to remove line breaks
    with open(train_record_json, "w") as f:
        json.dump(record, f, indent=2)
    # Response Function
    R = create_F()
    PSF = fspecial('gaussian', 8, 3)



    net=RASD_FuNet().cuda()


    # data
    train_data = FusionDataProcess(args.dataname,args.path,args.cache_path, R, args.patchsize, args.stride, args.factor, PSF,"train")
    train_loader = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,num_workers=8, worker_init_fn=seed_worker,pin_memory=True)

    test_data = FusionDataProcess(args.dataname,args.path,args.cache_path, R, args.patchsize, args.stride, args.factor, PSF,"test")
    test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False,num_workers=0, pin_memory=True)
    

    test_path=os.path.join(args.path,args.dataname,"test")

    maxiteration=len(train_data)//args.batch_size*args.epoch
    print(maxiteration)


    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxiteration, eta_min=1e-6, last_epoch=-1)

    loss_func=nn.L1Loss()


    best_psnr=0
    best_sam=0
    # train
    for epoch in range(1, args.epoch+1):
        net.train()
        trainloss = AverageMeter()
        loop = tqdm(train_loader, total=len(train_loader),ncols=100)
        for lrhsi, msi, gt in loop:
            lrhsi, msi, gt=lrhsi.cuda(), msi.cuda(), gt.cuda()

            optimizer.zero_grad()

            lr = optimizer.param_groups[0]['lr']

            fusion = net(lrhsi, msi)
            loss = loss_func(fusion, gt)

            trainloss.update(loss)
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{args.epoch}]')
            loop.set_postfix({'loss': '{0:1.8f}'.format(trainloss.avg.item()), "lr": '{0:1.8f}'.format(lr)})
            if args.clip_grad:
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.1)
            scheduler.step()
        
        record=[]
        if os.path.exists(train_record_json):
            with open(train_record_json, "r") as f:
                record = json.load(f)       
        record.append(
                    f"epoch:{epoch},"
                    f"loss:{trainloss.avg.item()},"
                    f"lr:{lr}")
        with open(train_record_json, "w") as f:
            json.dump(record, f, indent=2)

        # test
        if epoch==1 or epoch%args.test__freq==0:

            net.eval()
            val_loss=AverageMeter()
            sam = AverageMeter()
            psnr = AverageMeter()
            imglist = os.listdir(test_path)
            with torch.no_grad():
                for i,(lrhsi, msi, gt) in enumerate(test_loader):
                    # fusion = reconstruction(net, lrhsi.cuda(), msi.cuda(),args.factor, 64, 32)
                    # fusion = reconstruction_parallel(net, lrhsi.cuda(), msi.cuda(),args.factor, 64, 32,16)

                    fusion=net(lrhsi.cuda(),msi.cuda())
                    
                    fusion=torch.clamp(fusion,0,1)
                    val_loss.update(loss_func(fusion, gt.cuda()))
                    # print(Fuse.shape)
                    psnr_current=metrics.calc_psnr(fusion,gt.cuda())
                    sam_current=metrics.calc_sam(fusion.squeeze(0),gt.cuda().squeeze(0))
                    psnr.update(psnr_current)
                    sam.update(sam_current)

            torch.save(net.state_dict(),save_path_model_num +'/'+ str(epoch) + 'epoch' + '_.pkl')
            print("val  PSNR:",psnr.avg,  "  SAM:", sam.avg,"val loss:", val_loss.avg)
            record=[]
            if os.path.exists(test_record_json):
                with open(test_record_json, "r") as f:
                    record = json.load(f)
            record.append(
                f"epoch:{epoch}, "
                f"lr:{lr}, "
                f"trainloss:{trainloss.avg}, "
                f"val_loss:{val_loss.avg}, "
                f"psnr:{psnr.avg}, "
                f"sam:{sam.avg}")
            with open(test_record_json, "w") as f:
                json.dump(record, f, indent=2)      

        if  epoch == 0:
            torch.save(net.state_dict(),save_path_model_num +'/' + '_PSNR_best.pkl')
            best_psnr=psnr.avg
            best_sam=sam.avg


        elif best_psnr<psnr.avg:
            best_psnr=psnr.avg
            torch.save(net.state_dict(), save_path_model_num + '/' + '_PSNR_best.pkl')
        
        elif best_sam > sam.avg:
            best_sam=sam.avg
            torch.save(net.state_dict(), save_path_model_num + '/' + '_SAM_best.pkl')

        
        
       

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
