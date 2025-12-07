import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim


device = torch.device('cuda:0')


def normal(feat, eps=1e-5):
    feat_mean, feat_std = calc_mean_std(feat, eps)
    normalized = (feat-feat_mean)/feat_std
    return normalized


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.dist = lpips.LPIPS(net='alex')

    def forward(self, x, y):
        # images must be in range [-1, 1]
        dist = self.dist(2 * x - 1, 2 * y - 1)
        return dist


class losses(nn.Module):
    def __init__(self):
        super(losses, self).__init__()
        vgg = nn.Sequential(
                nn.Conv2d(3, 3, (1, 1)),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(3, 64, (3, 3)),
                nn.ReLU(),  # relu1-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),  # relu1-2
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 128, (3, 3)),
                nn.ReLU(),  # relu2-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),  # relu2-2
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 256, (3, 3)),
                nn.ReLU(),  # relu3-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  # relu3-2
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  # relu3-3
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  # relu3-4
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 512, (3, 3)),
                nn.ReLU(),  # relu4-1, this is the last layer used
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu4-2
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu4-3
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu4-4
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu5-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu5-2
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu5-3
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU()  # relu5-4
            )
        vgg.load_state_dict(torch.load("model_path/vgg_normalised.pth"))
        vgg = nn.Sequential(*list(vgg.children())[:44]).to(device)
        enc_layers = list(vgg.children())
        self.mse_loss = nn.MSELoss()
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
                for param in getattr(self, name).parameters():
                        param.requires_grad = False

        self.lpips = LPIPS().to(device)

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
                func = getattr(self, 'enc_{:d}'.format(i + 1))
                results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        mse_loss = nn.MSELoss()
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return mse_loss(input, target)

    def psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
        max_pixel = 1.0  # 假设像素值范围在 0 到 1 之间
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr

    def ssim(self, image1, image2):
        image1_np = image1.cpu().detach().numpy()
        image2_np = image2.cpu().detach().numpy()
        image1_np = np.transpose(image1_np, (0, 2, 3, 1))
        image2_np = np.transpose(image2_np, (0, 2, 3, 1))
        ssim_value = ssim(image1_np[0], image2_np[0], win_size=3, data_range=image1_np.max() - image1_np.min(), multichannel=True)
        return ssim_value


    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, content, style, gener):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        gener_feats = self.encode_with_intermediate(gener)

        loss_s = 0
        loss_c = 0
        lpips_out = 0
        psnr = self.psnr(content, gener)
        ssim = self.ssim(content, gener)
        for i in range(0, 5):
            loss_s += self.calc_style_loss(gener_feats[i], style_feats[i])

        for j in range(3, 5):
            loss_c += self.calc_content_loss(normal(gener_feats[j]), normal(content_feats[j]))
        lpips_out += torch.sum(self.lpips(gener, content))
        loss_c = loss_c / 2
        loss_s = loss_s / 5
        return loss_c, loss_s, lpips_out, psnr, ssim



