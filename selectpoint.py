import time
import torch
import torch.nn as nn

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x


def batch_index_fill_2(x, x1, idx1):
    B, N, C = x.size()
    B, N1, C = x1.size()


    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N


    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x = x.reshape(B, N, C)
    return x
    

if __name__ == '__main__':
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1" 

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    w=64
    hsi = torch.randn((2, 31,w//8,w//8), device=device)
    score=torch.randn((2, 1,w//8,w//8), device=device) 
    score=score.reshape(2,-1)
    sorted_indices = torch.argsort(score, dim=1, descending=True)
    idx1=sorted_indices[:,:25]
    idx2=sorted_indices[:,25:]
    
    v1= batch_index_select(hsi.reshape(2,31,-1).permute(0,2,1), idx1)
    v2= batch_index_select(hsi.reshape(2,31,-1).permute(0,2,1), idx2)
    out = batch_index_fill(hsi.reshape(2,31,-1).permute(0,2,1).clone(), v1, v2.clone(), idx1, idx2)
    print(out==hsi.reshape(2,31,-1).permute(0,2,1))
