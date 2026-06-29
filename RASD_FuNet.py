from einops import rearrange
from thop import profile, clever_format
import time
import torch
import torch.nn as nn

from ss2d_ import  IASM_Linear2_CA_pos_sort
from selectpoint import batch_index_fill, batch_index_select
# from ptflops import get_model_complexity_info


class MSA(nn.Module):
    def __init__(self, num_vector, num_heads_column, heads_number):
        """
        :param num_vector: Number of vectors
        :param num_heads_column: Number of columns in the Wk matrix—dim_vector × num_heads_column
        :param heads_number: Number of heads
        """
        super(MSA, self).__init__()
        # print(num_vector,num_heads_column,heads_number)
        self.num_vector = num_vector
        self.num_heads_column = num_heads_column
        self.heads_number = heads_number
        self.to_q = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.to_k = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.to_v = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads_number, 1, 1))  # Weight Parameter * CORE
        self.proj = nn.Linear(num_heads_column * heads_number, num_vector)
        # The location code here needs to be modified; do not use M-MSA.
        self.pos_emb = nn.Sequential(
            nn.Linear(num_heads_column * heads_number, num_vector),
            nn.modules.activation.GELU(),
            nn.Linear(num_vector, num_vector),
        )

    def forward(self, x_in):
        """
        :param x_in:
                    If the input image dimensions are b, h, w, c, then attention is applied to c.
                    If the input image dimensions are b, w, c, h, then attention is applied to h.
                    If the input image dimensions are b, c, h, w, then attention is applied to w.
        :return out_c:    The shape of the output from the backbone matches that of the input.
                out_p:    The shape of the position-encoded output matches that of the input.
                To make it easier to perform separate convolution operations later, we will not add them together.
        """
        b, n, c = x_in.shape
        # print (x_in.shape)
        x = x_in
        # print (x.shape)
        # print("to_q前{}".format(x.shape))
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # print(v_inp.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads_number),
                      (q_inp, k_inp, v_inp))
        v = v
        # q,k,v: b,heads,hw,c   The “c” here is no longer the “c” from `x_in.shape`; it should be `num_heads_column`.
        q = q.transpose(-2, -1)  # q,k,v: b,heads,c,hw
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = nn.functional.normalize(q, dim=-1, p=2)
        k = nn.functional.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, n, self.heads_number * self.num_heads_column)
        out_c = self.proj(x)  # Before the view, the shape of x is b, h*w, num_vector. num_vector is the c-dimension of x_in.shape.

        out_p = self.pos_emb(v_inp)

        return out_c+out_p

class Transformer(nn.Module):
    def __init__(self,x_channel):
        super(Transformer,self).__init__()
        self.saln1 = nn.LayerNorm(x_channel)
        self.saln2 = nn.LayerNorm(x_channel)
        self.sa=MSA(x_channel,x_channel,2)
        self.re_conv1=nn.Sequential(
            nn.Linear(x_channel,x_channel//2,bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(x_channel//2,x_channel,bias=False),
        )

    def forward(self,v1):
        nor_v1=self.saln1(v1)
        re_fea1=self.sa(nor_v1)+v1
        norre_fea1=self.saln2(re_fea1)
        refine1=self.re_conv1(norre_fea1)+re_fea1

        return refine1


class LRD(nn.Module):
    def __init__(self,  msi_bands=3,hsi_bands=31, dim=32,dim2=4):
        super(LRD, self).__init__()
        self.local_conv=nn.Sequential(nn.Conv2d(dim,dim2,1,1,0),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(dim2,dim2,3,1,1,groups=dim2)
                                      )
        
        self.global_conv=nn.Sequential(
                                nn.Conv2d(dim,dim2,1,1,0),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(dim2,dim2,3,1,1,dilation=1,groups=dim2),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(dim2,dim2,3,1,1,dilation=1,groups=dim2),
                                )
        self.out_score=nn.Sequential(
            nn.Conv2d(dim2*2,dim2,1,1,0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(dim2,1,1,1,0),
            nn.Sigmoid(),
        )

    def forward(self, fea):

        local_f=self.local_conv(fea)
        global_f=self.global_conv(fea)
        f=torch.cat([local_f,global_f],dim=1)
        pred_score=self.out_score(f)

        return pred_score
       

class RTGB(nn.Module):
    def __init__(self, dim=32):
        super(RTGB, self).__init__()


        self.conv1d_w = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv1d_h = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv1d_c = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        
        self.SA_conv=nn.Sequential(
            nn.Conv2d(1,1,3,1,1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1,1,3,1,1),
        )

        self.CA_linaer=nn.Sequential(
            nn.Linear(dim,dim//4),
            nn.LeakyReLU(0.1),
            nn.Linear(dim//4,dim)
        )


    def forward(self, x):
        B,C,W,H= x.shape
        vector_c=x.mean([2,3]).reshape(B,1,-1)
        vector_w=x.mean([1,3]).reshape(B,1,-1)
        vector_h=x.mean([1,2]).reshape(B,1,-1)

        vector_c=self.conv1d_c(vector_c)

        vector_w=self.conv1d_w(vector_w)

        vector_h=self.conv1d_h(vector_h)

        mask_wh = torch.einsum('bcw,bch->bcwh', vector_w, vector_h)  # shape (B, C, W, H)

        mask_wh=self.SA_conv(mask_wh)

        vector_c=self.CA_linaer(vector_c).permute(0,2,1).reshape(B,-1)
        
        out = torch.einsum('bc,bwh->bcwh', vector_c, mask_wh.squeeze(1))  # shape (B, C, W, H)

        return out


class CPDM(nn.Module):
    def __init__(self, msi_bands=3, hsi_bands=31,dim=32):
        super(CPDM, self).__init__()
       
        self.rtgb1 = RTGB(dim)
        self.rtgb2 = RTGB( dim)
        # self.rtgb3 = RTGB(msi_bands, hsi_bands, dim)

        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1,1,0),
        )

    def forward(self,x):
       
        o1 = self.rtgb1(x)  # x is B × C × W × H.  o1 is also    o2 is also
        o2 = self.rtgb2(x - o1)
        o_all = torch.cat((o1, o2), dim=1)
        output = self.conv(o_all)  # Learning the coefficients of each low-rank tensor in CP decomposition
        return output  # This is already a rank-1 tensor after CP decomposition.

        
        
class Mamba_Block(nn.Module):
    def __init__(self,dim=32):
        super(Mamba_Block, self).__init__()
        self.ln1=nn.LayerNorm(dim)

        self.asfcm=IASM_Linear2_CA_pos_sort(dim,dim//2,expand=2, dropout=0.3)
        
        # self.asfcm=IASM_Linear2_nosort_CA(dim,dim//2,expand=2, dropout=0.3)

        # self.asfcm=IASM_Linear_nosort(dim,dim//2,expand=2, dropout=0.3)

        # self.ln2=nn.LayerNorm(dim)
        # self.ffn=nn.Sequential(
        #     nn.Linear(dim,dim,bias=False),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(dim,dim,bias=False),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(dim,dim,bias=False),
        #     nn.LeakyReLU(0.1),
        # )

    def forward(self,x,attn,pos):
        x_nor=self.ln1(x)
        fea=self.asfcm(x_nor,attn,pos)+x
        # fea_nor=self.ln1(fea)
        # out=self.ffn(fea_nor)+fea
        return fea


class SLBlock(nn.Module):
    def __init__(self, msi_bands=3, hsi_bands=31,dim=32,windowsize=8):
        super(SLBlock, self).__init__()

        self.heads_number=2
        # self.prefusion=nn.Sequential(
        #         nn.Conv2d(dim,dim,1,1,1),
        #         nn.LeakyReLU(0.1)
        # )

        self.linaer_cat=nn.Linear(dim+msi_bands+hsi_bands,dim)
        self.lrd=LRD(msi_bands,hsi_bands,dim)
        self.cp=CPDM(msi_bands,hsi_bands,dim)

        self.asfcm=Mamba_Block(dim)

        # self.asfcm=WIASM_Conv_sort(dim,dim//2,windowsize,expand=1, dropout=0.3)
        # self.asfcm=WIASM_Conv_sort_2(dim,dim//2,windowsize,expand=1, dropout=0.3)
        # self.asfcm=SS2D(dim,dim//2,expand=1, dropout=0.3)
        # self.asfcm=WIASM_Conv_window_sort(dim,dim//2,windowsize,expand=1, dropout=0.3)

        # self.asfcm=IASM_Conv(dim,dim//2,expand=1, dropout=0.3)

        # self.ssm=ASFM(dim,dim//2,expand=2,dropout=0.3)
        # self.speT=Transformer(dim)

        self.to_q = nn.Linear(dim, self.heads_number*dim, bias=False)
        self.to_k = nn.Linear(dim, self.heads_number*dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(self.heads_number, 1, 1))

        self.conv_1=nn.Sequential(
            nn.Conv2d(dim*2,dim,1,1,0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim,dim,3,1,1,groups=dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim,dim,3,1,1,groups=dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim,dim,1,1,0)
        )

        # self.conv_1=nn.Sequential(
   
        #     nn.Conv2d(dim*3,dim,1,1,0)
        # )
        # self.conv_2=nn.Conv2d(dim,dim,1,1,0)

    def sobel_gradient(self, x):
        """
        Normalize the input first, then compute the gradient to improve numerical stability.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, W, H).
        
        Returns:
            grad_magnitude (torch.Tensor): Normalized gradient magnitude in [0,1].
            grad_x (torch.Tensor): Gradient along x-axis.
            grad_y (torch.Tensor): Gradient along y-axis.
        """
        # --------------------- Input Normalization ---------------------
        # Normalize each channel independently to the range [0,1]
        min_val = x.view(x.size(0), x.size(1), -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)  # (B,C,1,1)
        max_val = x.view(x.size(0), x.size(1), -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # (B,C,1,1)
        x_normalized = (x - min_val) / (max_val - min_val + 1e-6)  # (B,C,W,H)
        
        # ---------------------  Sobel Gradient Calculation ---------------------
        # Define the scaled-down Sobel operator (to prevent the response values from becoming too large)
        sobel_x = torch.tensor([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], 
                            dtype=x.dtype, device=x.device).view(1,1,3,3) / 8.0  # Scale to [-0.5, 0.5]
        
        sobel_y = torch.tensor([[-1, -2, -1], 
                                [ 0,  0,  0], 
                                [ 1,  2,  1]], 
                                dtype=x.dtype, device=x.device).view(1,1,3,3) / 8.0
        
        # Boundary Copy Fill (Using Normalized Input)
        x_padded = nn.functional.pad(x_normalized, (1,1,1,1), mode='replicate')  # (B,C,W+2,H+2)
        
        # Channel-wise Convolution (Supports Multi-Channel Inputs)
        grad_x = nn.functional.conv2d(
            x_padded, 
            sobel_x.expand(x.size(1), -1, -1, -1),  # Expand to (C, 1, 3, 3)
            groups=x.size(1)                         # Multichannel Convolution Processing in Groups
        )  # (B,C,W,H)
        
        grad_y = nn.functional.conv2d(
            x_padded,
            sobel_y.expand(x.size(1), -1, -1, -1),
            groups=x.size(1)
        )
        
        # --------------------- Calculation of Gradient Magnitude ---------------------
        # Add dual protection: negative number check + division-by-zero check
        # eps = 1e-6
        # grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + eps)  # Prevent sqrt(0)
        
        # # Normalize the entire dataset to the range [0,1] (to avoid the problem of excessively small denominators caused by channel-independent normalization).
        # global_min = grad_magnitude.min()                         # Scalar
        # global_max = grad_magnitude.max()                         # Scalar
        # grad_magnitude = (grad_magnitude - global_min) / (global_max - global_min + eps)
        
        # return grad_magnitude, grad_x, grad_y
        return  grad_x, grad_y

    
    def forward(self,hsi, msi,fea):
        
        res=fea
        # fea=self.prefusion(fea)+fea

        B,C,W,H=fea.shape

        # Predicted Score
        pred_score=self.lrd(fea)
        # mask=torch.nn.functional.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]
        mask_1=pred_score  # Low rank
        mask_2=1-pred_score

        sorted_indices = torch.argsort(pred_score.reshape(B,-1), dim=1, descending=True)
        # r=(1-self.info_content_normalized_tv(fea))
        r=torch.sqrt(torch.Tensor([0.5]))

        W_1=int(W*r)
        H_1=int(H*r)
        H_1=max(1,H_1)
        W_1=max(1,W_1)

        H_1=min(H-1,H_1)
        W_1=min(W-1,W_1)
        N=W_1*H_1
        
        idx_1=sorted_indices[:,:N]     # Low rank
        idx_2=sorted_indices[:,N:]    # Non-low-rank

        fea_1=fea*mask_1
        fea_2=fea*mask_2


        grad_x, grad_y=self.sobel_gradient(fea)

        # A_expanded = fea_2.unsqueeze(2)  # shape (B, C, 1, W, H)
        B_expanded = grad_x.unsqueeze(2)  # shape (B, C, 1, W, H)
        C_expanded = grad_y.unsqueeze(2)  # shape (B, C, 1, W, H)

        # Splicing along the new dimension yields the shape (B, C, 2, W, H)
        combined = torch.cat([B_expanded,C_expanded], dim=2)

        # Merge the channel dimensions to obtain the final shape (B, 2C, W, H)
        combined = combined.reshape(B, 2*C, W, H)
        # combined=self.position_emb(combined)
        

        v_1= batch_index_select(fea_1.reshape(B,C,-1).permute(0,2,1), idx_1)
        v_2= batch_index_select(fea_2.reshape(B,C,-1).permute(0,2,1), idx_2)
        # v_hsi= batch_index_select(scale_hsi.reshape(B,scale_hsi.shape[1],-1).permute(0,2,1), idx_2)

        v_c_2= batch_index_select(fea.reshape(B,C,-1).permute(0,2,1), idx_2)
        # v_hsi= batch_index_select(hsi.reshape(B,31,-1).permute(0,2,1), idx_2)
        combined= batch_index_select(combined.reshape(B,2*C,-1).permute(0,2,1), idx_2)



        q_inp = self.to_q(v_c_2)
        k_inp = self.to_k(v_c_2)
        q, k= map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads_number),
                      (q_inp, k_inp))

        # q,k,v: b,heads,hw,c   The “c” here is no longer the “c” from `x_in.shape`; it should be `num_heads_column`.
        q = q.transpose(-2, -1)  # q,k,v: b,heads,c,n
        k = k.transpose(-2, -1)
        q = nn.functional.normalize(q, dim=-1, p=2)
        k = nn.functional.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
       
        
        
        # Low-Rank Decomposition
        v_1=v_1.reshape(B,W_1,H_1,C).permute(0,3,1,2)

        v_1=self.cp(v_1)*v_1
        v_1=v_1.reshape(B,C,-1).permute(0,2,1)

        # Non-low-rank non-decomposition is fed directly into the `linear` layer, and later replaced with a Transformer.
        # v_2=v_2.reshape(B,-1,C)
        # v_dic=torch.cat([v_2,v_hsi],dim=-1)

        # spe_dic=self.to_E(v_dic)
        # spe_abu=self.to_A(v_2)
        v_2=self.asfcm(v_2,attn,combined)
        


        fea_all = batch_index_fill(fea.reshape(B,C,-1).permute(0,2,1).clone(), v_1, v_2, idx_1, idx_2)
        fea_all=fea_all.permute(0,2,1).reshape(B,C,W,H)
        fea_all=torch.cat([fea_all,fea],dim=1)
        # fea_all=torch.cat([fea_all,fea],dim=1)

        fea_all=self.conv_1(fea_all)

        out=fea_all+res
        return out
    

class MSLBlock(nn.Module):
    def __init__(self, msi_bands=3, hsi_bands=31,dim=32):
        super(MSLBlock, self).__init__()


        self.body_1=SLBlock(msi_bands, hsi_bands,dim,8)

        self.down_1 = nn.Sequential(
                    # 4 times
                    nn.Conv2d(dim, dim, kernel_size=6, stride=4, padding=2, bias=False),
                    # 2 times
                    nn.Conv2d(dim, dim, (4, 4), (2, 2), 1, bias=False)
                )
        
        self.body_2=SLBlock(msi_bands, hsi_bands,dim,8)

        self.up = nn.Sequential(
            nn.Conv2d(dim, dim*64, 1)
        )
        self.ps=nn.PixelShuffle(8)

        self.conv_cat=nn.Conv2d(dim*2,dim,1)

        # self.body_3=SLBlock(msi_bands, hsi_bands,dim)


    
    def forward(self,hsi,msi,fea,uphsi,donwmsi):
        fea_1=self.body_1(uphsi,msi,fea)
        fea_2=self.down_1(fea_1)
        fea_2=self.body_2(hsi,donwmsi,fea_2)
        fea_2=self.up(fea_2)
        fea_2=self.ps(fea_2)
        fea_3=torch.cat([fea_1,fea_2],dim=1)
        fea_3=self.conv_cat(fea_3)
        # out=self.body_3(hsi,msi,fea_3)
        out=fea_3
        return out
    


class RASD_FuNet(nn.Module):
    def __init__(self, msi_bands=3, hsi_bands=31,dim=32):
        super(RASD_FuNet, self).__init__()

        self.prefusion=nn.Conv2d(msi_bands+hsi_bands, dim, 1, 1, 0)

        self.body_1=MSLBlock(msi_bands, hsi_bands,dim)

        self.body_2=MSLBlock(msi_bands, hsi_bands,dim)

        self.body_3=MSLBlock(msi_bands, hsi_bands,dim)

        self.body_4=MSLBlock(msi_bands, hsi_bands,dim)

        self.body_5=MSLBlock(msi_bands, hsi_bands,dim)

        self.refine=nn.Sequential(
            nn.Conv2d(dim*6,hsi_bands,1, 1, 0, bias=False)
        )
    
    def forward(self, hsi,msi):
        uphsi = torch.nn.functional.interpolate(hsi, scale_factor= 8, mode='bicubic')
        donwmsi=torch.nn.functional.interpolate(msi, scale_factor= 0.125, mode='bicubic')
        fea=torch.cat((uphsi,msi),dim=1)
        fea=self.prefusion(fea)


        fea_1=self.body_1(hsi,msi,fea,uphsi,donwmsi)

        fea_2=self.body_2(hsi,msi,fea_1,uphsi,donwmsi)

        fea_3=self.body_3(hsi,msi,fea_2,uphsi,donwmsi)

        fea_4=self.body_4(hsi,msi,fea_3,uphsi,donwmsi)

        fea_5=self.body_5(hsi,msi,fea_4,uphsi,donwmsi)

        refine=self.refine(torch.cat((fea_5,fea_4,fea_3,fea_2,fea_1,fea),dim=1))
        out=refine + uphsi
        return out


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    print("*" * 10 + "The HSRnet.py file is running" + "*" * 10)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device="cpu"
    loss_func=nn.L1Loss().to(device)

    hsi = torch.randn((1, 31,512// 8, 512//8), device=device)
    msi = torch.randn((1,3, 512, 512), device=device)
    gt = torch.randn((1, 31,512, 512), device=device)

    # hsi = torch.randn((1, 31,64// 8, 64//8), device=device)
    # msi = torch.randn((1,3, 64, 64), device=device)
    # gt = torch.randn((1, 31,64, 64), device=device)

    model= RASD_FuNet(3,31,32).to(device)
    # model.train()

    # out = model(hsi, msi)
    # print(out.shape)
    # loss=loss_func(out,gt)
    # loss.backward()

    flops, params = profile(model, inputs=(hsi,msi))
    print(flops, params)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    params=sum(p.numel() for p in model.parameters())
    print(params/1000000)


    # input_size1 = (3, 512, 512)  
    # # Use `get_model_complexity_info` to view the model's complexity
    # flops, params = get_model_complexity_info(model, (input_size1), as_strings=True, print_per_layer_stat=True)
    # print("%s |%s" % (flops, params))

    for i in range(10):
        model.eval()
        with torch.no_grad():
            time1 = time.time()
            out = model(hsi, msi)
            time2 = time.time()
            print(time2 - time1,out.shape)    






            










    


