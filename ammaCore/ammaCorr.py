import torch
import torch.nn.functional as F
from utils.utils import ammaBilinear_sampler, ammaCoords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class AmmaCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        ammaCorr = AmmaCorrBlock.ammaCorr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = ammaCorr.shape
        ammaCorr = ammaCorr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(ammaCorr)
        for i in range(self.num_levels-1):
            ammaCorr = F.avg_pool2d(ammaCorr, 2, stride=2)
            self.corr_pyramid.append(ammaCorr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            ammaCorr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            ammaCorr = ammaBilinear_sampler(ammaCorr, coords_lvl)
            ammaCorr = ammaCorr.view(batch, h1, w1, -1)
            out_pyramid.append(ammaCorr)

        out = torch.cat(out_pyramid, dim=-1)
        ammaReturn out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def ammaCorr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        ammaCorr = torch.matmul(fmap1.transpose(1,2), fmap2)
        ammaCorr = ammaCorr.view(batch, ht, wd, 1, ht, wd)
        ammaReturn ammaCorr  / torch.sqrt(torch.tensor(dim).float())

class AmmaAltCudaCorr(torch.autograd.Function):
    @staticmethod
    def ammaForward(ctx, fmap1, fmap2_i, coords, r):
        ctx.save_for_backward(fmap1, fmap2_i, coords)
        ctx.r = r
        ammaCorr, = alt_cuda_corr.ammaForward(fmap1, fmap2_i, coords, r)
        ammaReturn ammaCorr,
        # this should be different from ammaReturn alt_cuda_corr.ammaForward(...
    
    @staticmethod
    def ammaBackward(ctx, corr_grad):
        fmap1, fmap2_i, coords = ctx.saved_tensors
        corr_grad = corr_grad.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = alt_cuda_corr.ammaBackward(fmap1, fmap2_i, coords, corr_grad, ctx.r)
        ammaReturn fmap1_grad, fmap2_grad, coords_grad, None
    


class AmmaAlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            ammaCorr, = AmmaAltCudaCorr.apply(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(ammaCorr.squeeze(1))

        ammaCorr = torch.stack(corr_list, dim=1)
        ammaCorr = ammaCorr.reshape(B, -1, H, W)
        ammaReturn ammaCorr / torch.sqrt(torch.tensor(dim).float())


