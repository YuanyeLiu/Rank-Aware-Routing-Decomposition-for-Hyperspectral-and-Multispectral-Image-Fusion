import torch
import torch.nn.functional as F


def calc_rmse(fused, ref):
    fused=fused*255
    ref=ref*255
    mse = torch.mean((fused - ref) ** 2)
    rmse = mse.sqrt()
    return rmse.item()


def calc_psnr(fused, ref, max_val=1.):
    mse = torch.mean((fused - ref) ** 2, [0,2,3])
    psnr = 10*torch.mean(torch.log10(max_val / mse))
    return psnr.item()


def calc_sam(im1, im2, eps=2.2204e-16):
    """
    im1, im2: torch.Tensor, CHW format
    Output: scalar SAM in degrees, equivalent to MATLAB HWC implementation
    """
    # force double precision
    im1 = im1.double()*255
    im2 = im2.double()*255

    # CHW -> HWC
    im1 = im1.permute(1, 2, 0)
    im2 = im2.permute(1, 2, 0)

    H, W, C = im1.shape

    # reshape (H*W, C)
    im1 = im1.reshape(-1, C)
    im2 = im2.reshape(-1, C)

    # dot product and norms
    mole = torch.sum(im1 * im2, dim=1)
    norm1 = torch.sqrt(torch.sum(im1**2, dim=1))
    norm2 = torch.sqrt(torch.sum(im2**2, dim=1))

    # MATLAB style (a+eps)/(b+eps)/(c+eps)
    tmp = (mole + eps) / (norm1 + eps) / (norm2 + eps)

    # acos + real, handle NaN like MATLAB
    sam = torch.acos(tmp)
    sam = torch.where(torch.isnan(sam), torch.zeros_like(sam), sam)

    # convert rad -> deg
    sam = sam * 180 / torch.pi

    return sam.mean().item()



def gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu"):
    """ 11×11 """
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = g[:, None] * g[None, :]   # outer product
    window = window / window.sum()
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def calc_ssim(img1, img2, window_size=11, sigma=1.5, K=(0.01, 0.03), L=1):
    """
    img1, img2: torch.Tensor, shape (N, C, H, W)
    L: Dynamic range (e.g., 1.0 or 255)
    """
    device = img1.device
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    N, C, H, W = img1.shape
    window = gaussian_window(window_size, sigma, C, device=device)

    # μ
    mu1 = F.conv2d(img1, window, padding=0, groups=C)
    mu2 = F.conv2d(img2, window, padding=0, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # σ
    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=C) - mu1_mu2

    # SSIM map
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    # Calculate the mean of H and W, then calculate the mean of C.
    ssim = ssim_map.mean(dim=[2, 3]).mean(dim=1)  # (N,)

    return ssim.item()





def calc_uiqi(img1, img2, block_size=32):
    """
    img1, img2: (N, C, H, W) or (C,H,W) or (H,W)
    return:
        quality: scalar (mean UIQI)
        quality_map: (N,C,H-block+1, W-block+1)
    """

    # ---- reshape to (N,C,H,W) ----
    if img1.max()<=1:
        img1=img1*255
        img2=img2*255
    if img1.ndim == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)   # 1×1×H×W
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.ndim == 3:
        img1 = img1.unsqueeze(0)                # 1×C×H×W
        img2 = img2.unsqueeze(0)

    N, C, H, W = img1.shape

    # ---- block filter: sum2_filter ----
    kernel = torch.ones((C, 1, block_size, block_size), device=img1.device)

    # ---- required squared and product ----
    img1_sq = img1 * img1
    img2_sq = img2 * img2
    img12   = img1 * img2

    # ---- sum inside block ----
    img1_sum     = F.conv2d(img1,     kernel, stride=1, padding=0, groups=C)
    img2_sum     = F.conv2d(img2,     kernel, stride=1, padding=0, groups=C)
    img1_sq_sum  = F.conv2d(img1_sq,  kernel, stride=1, padding=0, groups=C)
    img2_sq_sum  = F.conv2d(img2_sq,  kernel, stride=1, padding=0, groups=C)
    img12_sum    = F.conv2d(img12,    kernel, stride=1, padding=0, groups=C)

    Np = block_size * block_size   # number of pixels per block

    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum

    numerator = 4 * (Np * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = Np * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul

    # ---- same branching as MATLAB ----
    quality_map = torch.ones_like(denominator)

    idx = (denominator1 == 0) & (img12_sq_sum_mul != 0)
    quality_map[idx] = 2 * img12_sum_mul[idx] / img12_sq_sum_mul[idx]

    idx = (denominator != 0)
    quality_map[idx] = numerator[idx] / denominator[idx]

    # ---- final mean over spatial dimension ----
    quality = quality_map.mean().item()

    return quality, quality_map


def compute_rmse_ergas(x, y, ratio_ergas):
    if x.max()<=1:
        x=x*255
        y=y*255
    # RMSE
    n_samples = x.shape[2] * x.shape[3]  # H * W
    n_bands = x.shape[1]  # C
    aux = torch.sum((x - y) ** 2, dim=(2, 3)) / n_samples  # Sum over height and width (assuming BCHW shape)
    rmse_per_band = torch.sqrt(aux)

    # ERGAS
    mean_y = torch.sum(y, dim=(2, 3)) / n_samples  # Sum over height and width (assuming BCHW shape)
    ergas = 100 * ratio_ergas * torch.sqrt(torch.sum((rmse_per_band / mean_y) ** 2) / n_bands)

    return ergas.item()