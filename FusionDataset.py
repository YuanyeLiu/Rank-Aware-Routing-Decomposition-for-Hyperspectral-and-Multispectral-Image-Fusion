import pickle
from torch.utils.data import Dataset
import hdf5storage as h5
import torch
import numpy as np
import os
from scipy import signal
import tqdm



def Gaussian_downsample(x, psf, s):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    y = np.zeros((x.shape[0], int(x.shape[1] / s), int(x.shape[2] / s)))
    for i in range(x.shape[0]):
        x1 = x[i, :, :]
        x2 = signal.convolve2d(x1, psf, boundary='symm', mode='same')
        y[i, :, :] = x2[0::s, 0::s]
    return y


    
def crop_to_patch(img, size, stride):
    H, W = img.shape[:2]
    patches = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            if h + size <= H and w + size <= W:
                patch = img[h: h + size, w: w + size, :]
                patches.append(patch)
    return patches



class FusionDataProcess(Dataset):
    def __init__(self, dataname,path,cache_path, R, training_size, stride, downsample_factor, PSF, data_type):
        """
        :param path:
        :param R: spectral response matrix
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: Gaussian blur kernel
        :param data_type: train or test
        """
        cache_path = os.path.join(cache_path, data_type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.lr,self.hr,self.msi = [], [],[]
            datapath=os.path.join(path, dataname,data_type)
            print("Cache file not found. Generate it from: ", datapath)
            imglist = os.listdir(datapath)
        
            for hrms_name in tqdm.tqdm(imglist):
                if dataname == "CAVE":
                    hrms = h5.loadmat(os.path.join(datapath, hrms_name))["b"]
                if dataname == "KAIST":
                    hrms = h5.loadmat(os.path.join(datapath, hrms_name))["HSI"]
                    hrms=hrms/hrms.max()
                    height, width = hrms.shape[:2]
                    left = (width - 2048) // 2
                    top = (height - 2048) // 2
                    right = left + 2048
                    bottom = top + 2048
                    # crop images
                    hrms = hrms[top:bottom, left:right,:]

                elif dataname == "ICVL":
                    hrms = h5.loadmat(os.path.join(datapath, hrms_name))["rad"]
                    hrms = np.rot90(hrms)
                    hrms=hrms/hrms.max()
                w, h = int(hrms.shape[0] //downsample_factor), int(hrms.shape[1]//downsample_factor)
                hrms = hrms[ :w *downsample_factor, :h * downsample_factor,:]

                HRHSI = hrms# hwc
                HSI_LR = Gaussian_downsample(np.transpose(hrms, (2, 0, 1)), PSF, downsample_factor)
                HSI_LR=np.transpose(HSI_LR,(1,2,0))
                MSI = np.tensordot(HRHSI,R, axes=([2], [1]))


                HRHSI=HRHSI.astype(np.float16)  
                HSI_LR=HSI_LR.astype(np.float16)
                MSI=MSI.astype(np.float16) 

                if data_type == "train":
                    hr_patches=crop_to_patch(HRHSI, training_size,stride)
                    lr_patches=crop_to_patch(HSI_LR,training_size//downsample_factor,stride//downsample_factor)
                    msi_patches=crop_to_patch(MSI,training_size,stride)
                    self.lr+=lr_patches
                    self.hr+=hr_patches
                    self.msi+=msi_patches
                
                elif data_type =="test":
                    self.lr.append(HSI_LR)
                    self.hr.append(HRHSI)
                    self.msi.append(MSI)

            with open(cache_path, "wb") as f:
                pickle.dump([self.lr,self.hr,self.msi], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.lr,self.hr,self.msi = pickle.load(f)

    def __getitem__(self, index):
        train_hrhs = self.hr[index]
        train_hrms = self.msi[index]
        train_lrhs = self.lr[index]


        train_hrhs = torch.from_numpy(train_hrhs.astype(np.float32))
        train_hrhs = train_hrhs.permute(2, 0, 1)

        train_hrms = torch.from_numpy(train_hrms.astype(np.float32))
        train_hrms = train_hrms.permute(2, 0, 1)

        train_lrhs = torch.from_numpy(train_lrhs.astype(np.float32))
        train_lrhs = train_lrhs.permute(2, 0, 1)

        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_lrhs, train_hrms,train_hrhs

    def __len__(self):
        return len(self.hr)



class FusionDataProcess_remote(Dataset):
    def __init__(self, dataname,path,cache_path, R, training_size, stride, downsample_factor, PSF, data_type):
        """
        :param path:
        :param R: spectral response matrix
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: Gaussian blur kernel
        :param data_type: train or test
        
        """
        cache_path = os.path.join(cache_path, data_type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.raw, self.pan, self.hrms = [],[],[]
            base_path = os.path.join(path, dataname)
            print("Cache file not found. Generate it from: ", base_path)
            
            spe_res = np.array([[2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3]])
            spe_res=spe_res/spe_res.sum()

            
            if dataname == "Houston":
                hrms = h5.loadmat(os.path.join(base_path, "HSI.mat"))['HSI']
                hrms=hrms/hrms.max()

            w, h = int(hrms.shape[0] // downsample_factor//downsample_factor), int(hrms.shape[1] // downsample_factor//downsample_factor)
            h_half=h//2
            if type == "train":
                hrms = hrms[:w *  downsample_factor*downsample_factor , :h_half *  downsample_factor*downsample_factor,:]
            elif type == "test":
                hrms = hrms[:w *   downsample_factor*downsample_factor , h_half*   downsample_factor*downsample_factor: h *  downsample_factor*downsample_factor,:]



            HRHSI = hrms# hwc
            HSI_LR = Gaussian_downsample(np.transpose(hrms, (2, 0, 1)), PSF, downsample_factor)
            HSI_LR=np.transpose(HSI_LR,(1,2,0))
            MSI = np.tensordot(HRHSI,R, axes=([2], [1]))


            HRHSI=HRHSI.astype(np.float16)  
            HSI_LR=HSI_LR.astype(np.float16)
            MSI=MSI.astype(np.float16) 

            if data_type == "train":
                hr_patches=crop_to_patch(HRHSI, training_size,stride)
                lr_patches=crop_to_patch(HSI_LR,training_size//downsample_factor,stride//downsample_factor)
                msi_patches=crop_to_patch(MSI,training_size,stride)
                self.lr+=lr_patches
                self.hr+=hr_patches
                self.msi+=msi_patches
            
            elif data_type =="test":
                self.lr.append(HSI_LR)
                self.hr.append(HRHSI)
                self.msi.append(MSI)

            with open(cache_path, "wb") as f:
                pickle.dump([self.lr,self.hr,self.msi], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.lr,self.hr,self.msi = pickle.load(f)

    def __getitem__(self, index):
        train_hrhs = self.hr[index]
        train_hrms = self.msi[index]
        train_lrhs = self.lr[index]


        train_hrhs = torch.from_numpy(train_hrhs.astype(np.float32))
        train_hrhs = train_hrhs.permute(2, 0, 1)

        train_hrms = torch.from_numpy(train_hrms.astype(np.float32))
        train_hrms = train_hrms.permute(2, 0, 1)

        train_lrhs = torch.from_numpy(train_lrhs.astype(np.float32))
        train_lrhs = train_lrhs.permute(2, 0, 1)

        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_lrhs, train_hrms,train_hrhs

    def __len__(self):
        return len(self.hr)