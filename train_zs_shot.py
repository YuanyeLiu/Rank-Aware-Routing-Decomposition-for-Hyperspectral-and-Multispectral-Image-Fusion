import argparse
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  
        os.makedirs(path) 
        print("The training folder is:{}".format(path))
    else:
        print('Already exists{}'.format(path))


def Gaussian_downsample(x, psf, s):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    y = np.zeros((x.shape[0], int(x.shape[1] / s), int(x.shape[2] / s)))
    for i in range(x.shape[0]):
        x1 = x[i, :, :]
        x2 = signal.convolve2d(x1, psf, boundary='symm', mode='same')
        y[i, :, :] = x2[0::s, 0::s]
    return y


def Gaussian_downsample_torch(x, psf, s, device='cuda'):
    """
    x: NumPy array, shape (H, W), (C, H, W), or (B, C, H, W)
    psf: 2D NumPy array, convolution kernel
    s: downsampling stride
    device: ‘cuda’ or 'cpu'
    """
    # 转 torch
    if x.ndim == 2:
        x = x[None, None, :, :]  # (1,1,H,W)
    elif x.ndim == 3:
        x = x[None, :, :, :]     # (1,C,H,W)
    elif x.ndim == 4:
        pass  # (B,C,H,W)
    else:
        raise ValueError("x must be 2D, 3D, or 4D numpy array")

    x = torch.from_numpy(x).float().to(device)

    psf = torch.from_numpy(psf).float().to(device)
    kH, kW = psf.shape
    C = x.shape[1]

    psf = psf.unsqueeze(0).unsqueeze(0)       # (1,1,kH,kW)
    psf = psf.repeat(C, 1, 1, 1)              # (C,1,kH,kW)

    y = torch.nn.functional.conv2d(x, psf, padding=(kH//2, kW//2), groups=C)

    y = y[:, :, ::s, ::s]

    return y[0].cpu().numpy()


class RealDATAProcess_ZS(Dataset):
    def __init__(self, hsi,msi, training_size, stride, downsample_factor, PSF):
        """
        :param path:
        :param R: spectral response matrix
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: Gaussian blur kernel

        """
        train_hrhs = []
        train_lrhs = []
        train_hrms = []

        # hwc-chw
        HRHSI = np.transpose(hsi, (2, 0, 1))
        msi = np.transpose(msi, (2, 0, 1))

        HSI_LR = Gaussian_downsample_torch(HRHSI, PSF, downsample_factor)
        MSI = Gaussian_downsample_torch(msi, PSF, downsample_factor)



        for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
            for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                # if (j+training_size)>800 and k<400:
                #     pass
                # else:
                temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                temp_hrms = MSI[:, j:j + training_size, k:k + training_size]

                temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                            int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                # temp_hrhs=temp_hrhs.astype(np.float16)
                # temp_lrhs=temp_lrhs.astype(np.float16)
                # temp_hrms = temp_hrms.astype(np.float16)
                train_hrhs.append(temp_hrhs)
                train_lrhs.append(temp_lrhs)
                train_hrms.append(temp_hrms)

        train_hrhs = torch.from_numpy(np.array(train_hrhs))
        train_lrhs = torch.from_numpy(np.array(train_lrhs))
        train_hrms = torch.from_numpy(np.array(train_hrms))

        # train_hrhs = torch.from_numpy(np.array(train_hrhs,dtype=np.float16))
        # train_lrhs = torch.from_numpy(np.array(train_lrhs,dtype=np.float16))
        # train_hrms = torch.from_numpy(np.array(train_hrms,dtype=np.float16))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_lrhs_all = train_lrhs
        self.train_hrms_all = train_hrms

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]
    

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
    return HSI_recon



def main(args):
    
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
    # 
    with open(train_record_json, "w") as f:
        json.dump(record, f, indent=2)

   
   
    HSI= np.load(args.hsi_path)   
    MSI= np.load(args.msi_path)    
    R = np.load(args.R_path)
    C = np.load(args.C_path)
    R = np.transpose(R, (1, 0))


    for band in range(R.shape[0]):
        div = np.sum(R[band][:])
        for i in range(R.shape[1]):
            R[band][i] = R[band][i] / div

    HSI_LR = Gaussian_downsample(np.transpose(HSI, (2, 0, 1)),  C, args.factor)
    LRMSI = Gaussian_downsample(np.transpose(MSI, (2, 0, 1)),  C, args.factor)

    test_HRHSI0=np.transpose(HSI,(2, 0, 1))
    test_HRMSI0=LRMSI
    test_LRHSI0=HSI_LR


    print("Training data processing complete")


    loss_func = nn.L1Loss(reduction='mean').cuda()


   
    
    train_data=RealDATAProcess_ZS(HSI,MSI,args.patchsize, args.stride, args.factor,C)
    train_loader = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    
    maxiteration=len(train_data)//args.batch_size*args.epoch

    print("maxiteration：", maxiteration)


    cnn=RASD_FuNet(4,150,128).cuda()

    for m in cnn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr,betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10,20,30], 0.5)


    start_epoch = 0



    for epoch in range(start_epoch+1, args.epoch+1):
        cnn.train()
        trainloss = AverageMeter()
        loop = tqdm(train_loader, total=len(train_loader),ncols=100)
        for i,(a1, a2, a3) in enumerate(loop):
            a1=a1.float()
            a2=a2.float()
            a3=a3.float()
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()      

            output = cnn(a3.cuda(), a2.cuda())   # hsrnet
            loss = loss_func(output, a1.cuda())
        
            trainloss.update(loss)
        
            # optimizer.zero_grad()
            loss.backward()

            if i % 50 == 0:  # Print once every 100 batches to prevent the screen from being flooded with messages
                for name, param in cnn.named_parameters():
                    if param.grad is not None:
                        print(f"{name:30s} grad mean: {param.grad.abs().mean().item():.6e}")
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{args.epoch}]')
            loop.set_postfix({'loss': '{0:1.8f}'.format(trainloss.avg.item()), "lr": '{0:1.8f}'.format(lr)})
            if args.clip_grad:
                torch.nn.utils.clip_grad_value_(cnn.parameters(), clip_value=0.1)
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


        if epoch==1 or epoch%args.test__freq==0:
            cnn.eval()
            val_loss=AverageMeter()
            sam = AverageMeter()
            psnr = AverageMeter()

            with torch.no_grad():
                # img1 = img1 / img1.max()
                test_HRHSI = torch.unsqueeze(torch.Tensor(test_HRHSI0),0)
                test_HRMSI =torch.unsqueeze(torch.Tensor(test_HRMSI0),0)
                test_LRHSI=torch.unsqueeze(torch.Tensor(test_LRHSI0),0)


                fusion=reconstruction_fg5_parallel(cnn, test_LRHSI.cuda(), test_HRMSI.cuda(),args.factor, 64, 60,16)
                # print(Fuse.shape)
                fusion=torch.round(fusion*255)/255.0

                val_loss.update(loss_func(fusion.unsqueeze(0).cuda(),test_HRHSI.cuda()))
                psnr_current=metrics.calc_psnr(fusion.unsqueeze(0).cuda(),test_HRHSI.cuda())
                sam_current=metrics.calc_sam(fusion.cuda(),test_HRHSI.cuda().squeeze(0))
                psnr.update(psnr_current)
                sam.update(sam_current)

            torch.save(cnn.state_dict(),save_path_model_num +'/'+ str(epoch) + 'epoch' + '_.pkl')
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


        if  epoch == 1 or epoch == args.epoch:
            torch.save(cnn.state_dict(),save_path_model_num +'/' + '_PSNR_best.pkl')





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
