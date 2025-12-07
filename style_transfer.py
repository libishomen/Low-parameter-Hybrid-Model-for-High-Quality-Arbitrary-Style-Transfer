import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda:1"


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class SCI(nn.Module):
    def __init__(self, dim):
        super(SCI, self).__init__()
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1)
        )

        self.conv_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, Fc, Fcs):
        channel_map = self.channel_interaction(Fc)

        spatial_map = self.spatial_interaction(Fcs)

        # S-I
        Fcs = Fcs * torch.sigmoid(channel_map)
        # C-I
        Fc = Fc * torch.sigmoid(spatial_map)

        x = Fcs + Fc
        x = self.conv_out(x)
        return x


class SANet(nn.Module):
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h_c, w_c = F.size()
        F = F.view(b, -1, w_c * h_c).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        H = H.view(b, -1, w * h).transpose(1, 2)
        mean = torch.bmm(S, H)
        std = torch.sqrt(torch.relu(torch.bmm(S, H ** 2) - mean ** 2))
        mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
        out = std * mean_variance_norm(content) + mean
        return out


class CANet(nn.Module):
    def __init__(self, in_dim):
        super(CANet, self).__init__()
        self.J = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.K = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.W = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.R = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, content, style):
        Fc_tilde = self.J(mean_variance_norm(content))
        B, C, H, W = style.size()
        Fs_tilde = self.K(mean_variance_norm(style)).view(B, C, H * W)

        Gram_sum = Fs_tilde.sum(-1).view(B, C, 1)

        Gram_s = (Fs_tilde @ Fs_tilde.permute(0, 2, 1) / Gram_sum).view(B, C, C, 1)

        # Weight Gram Matrix
        Weighted_Gram = self.W(Gram_s).view(B, C, C)

        # get C weighted value
        sigma = torch.diagonal(Weighted_Gram, dim1=-2, dim2=-1).view(B, C, 1, 1)
        Fcs = self.R(Fc_tilde * sigma + content)
        return Fcs


class style_transfer(nn.Module):
    def __init__(self, dim):
        super(style_transfer, self).__init__()
        self.sanet = SANet(dim)
        self.canet = CANet(dim)
        self.fusion = SCI(dim)

    def forward(self, content, style):
        ca_Fcs = self.canet(content, style)
        sa_Fcs = self.sanet(content, style)
        output = self.fusion(ca_Fcs, sa_Fcs)
        out = output + content
        return out


if __name__ == '__main__':
    content = torch.rand(1, 256, 64, 64)
    style = torch.rand(1, 256, 64, 64)
    model = style_transfer(256)
    output = model(content, style)
    print(output.shape)
