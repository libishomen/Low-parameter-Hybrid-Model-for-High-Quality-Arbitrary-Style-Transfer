import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from function import calc_mean_std
from generate import NextVit, decoder
# from style_transfer_2 import TransModule
from style_transfer import style_transfer
from function import nor_mean_std
import numpy as np
from hist_loss import RGBuvHistBlock
import kornia


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

mlp = nn.ModuleList([nn.Linear(64, 64),
                     nn.ReLU(),
                     nn.Linear(64, 16),
                     nn.Linear(128, 128),
                     nn.ReLU(),
                     nn.Linear(128, 32),
                     nn.Linear(256, 256),
                     nn.ReLU(),
                     nn.Linear(256, 64),
                     nn.Linear(512, 512),
                     nn.ReLU(),
                     nn.Linear(512, 128)])


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class CCPL(nn.Module):
    def __init__(self, mlp):
        super(CCPL, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mlp = mlp

    def NeighborSample(self, feat, layer, num_s, sample_ids=[]):
        b, c, h, w = feat.size()
        feat_r = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if sample_ids == []:
            dic = {0: -(w + 1), 1: -w, 2: -(w - 1), 3: -1, 4: 1, 5: w - 1, 6: w, 7: w + 1}
            s_ids = torch.randperm((h - 2) * (w - 2), device=feat.device)  # indices of top left vectors
            s_ids = s_ids[:int(min(num_s, s_ids.shape[0]))]
            ch_ids = (s_ids // (w - 2) + 1)  # centors
            cw_ids = (s_ids % (w - 2) + 1)
            c_ids = (ch_ids * w + cw_ids).repeat(8)
            delta = [dic[i // num_s] for i in range(8 * num_s)]
            delta = torch.tensor(delta).to(feat.device)
            n_ids = c_ids + delta
            sample_ids += [c_ids]
            sample_ids += [n_ids]
        else:
            c_ids = sample_ids[0]
            n_ids = sample_ids[1]
        feat_c, feat_n = feat_r[:, c_ids, :], feat_r[:, n_ids, :]
        feat_d = feat_c - feat_n
        for i in range(3):
            feat_d = self.mlp[3 * layer + i](feat_d)
        feat_d = Normalize(2)(feat_d.permute(0, 2, 1))
        return feat_d, sample_ids

    ## PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation
    def PatchNCELoss(self, f_q, f_k, tau=0.07):
        # batch size, channel size, and number of sample locations
        B, C, S = f_q.shape
        ###
        f_k = f_k.detach()
        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S, dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(f_q.device)
        return self.cross_entropy_loss(predictions, targets)

    def forward(self, feats_q, feats_k, num_s, start_layer, end_layer, tau=0.07):
        loss_ccp = 0.0
        for i in range(start_layer, end_layer):
            f_q, sample_ids = self.NeighborSample(feats_q[i], i, num_s, [])
            f_k, _ = self.NeighborSample(feats_k[i], i, num_s, sample_ids)
            loss_ccp += self.PatchNCELoss(f_q, f_k, tau)
        return loss_ccp


class Net(nn.Module):
    def __init__(self, encoder, training_mode='art'):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        self.SCT = style_transfer(256)

        self.mlp = mlp if training_mode == 'art' else mlp[:9]

        self.CCPL = CCPL(self.mlp)

        self.mse_loss = nn.MSELoss()

        self.end_layer = 4 if training_mode == 'art' else 3
        self.mode = training_mode

        self.style_encoder = NextVit()  # 风格编码器
        self.content_encoder = NextVit()  # 内容编码器

        self.decoder = decoder()  # 解码器

        self.MSELoss = nn.MSELoss()

        self.hist = RGBuvHistBlock(insz=64, h=256,
                                   intensity_scale=True,
                                   method='inverse-quadratic')  # 计算color loss会使用到的

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_histogram_loss(self, A, B, histogram_block):
        input_hist = histogram_block(A)
        target_hist = histogram_block(B)
        histogram_loss = (1 / np.sqrt(2.0) * (torch.sqrt(torch.sum(
            torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) /
                          input_hist.shape[0])

        return histogram_loss

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(self.end_layer):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def feature_compress(self, feat):
        feat = feat.flatten(2, 3)
        feat = self.mlp(feat)
        feat = feat.flatten(1, 2)
        feat = Normalize(2)(feat)
        return feat

    def calc_content_loss(self, input, target):
        return self.mse_loss(nor_mean_std(input), nor_mean_std(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def cosine_dismat(self, A, B):
        A = A.view(A.shape[0], A.shape[1], -1)
        B = B.view(B.shape[0], B.shape[1], -1)

        A_norm = torch.sqrt((A ** 2).sum(1))
        B_norm = torch.sqrt((B ** 2).sum(1))

        A = (A / A_norm.unsqueeze(dim=1).expand(A.shape)).permute(0, 2, 1)
        B = (B / B_norm.unsqueeze(dim=1).expand(B.shape))
        dismat = 1. - torch.bmm(A, B)
        return dismat

    def calc_remd_loss(self, A, B):
        C = self.cosine_dismat(A, B)
        m1, _ = C.min(1)
        m2, _ = C.min(2)
        remd = torch.max(m1.mean(), m2.mean())
        return remd

    def S_T(self, content, style):
        c_f = self.content_encoder(content)
        s_f = self.style_encoder(style)
        cs_f = self.SCT(c_f, s_f)
        gimage = self.decoder(cs_f)
        return gimage

    def forward(self, content, style, num_s, num_layer):
        c_f = self.content_encoder(content)
        s_f = self.style_encoder(style)

        cs_f = self.SCT(c_f, s_f)
        gimage = self.decoder(cs_f)

        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)

        g_t_feats = self.encode_with_intermediate(gimage)

        end_layer = self.end_layer

        # content loss
        loss_c = self.calc_content_loss(g_t_feats[-1], content_feats[-1]) + self.calc_content_loss(g_t_feats[-2], content_feats[-2])

        # style loss
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        # ccp loss
        start_layer = end_layer - num_layer
        loss_ccp = self.CCPL(g_t_feats, content_feats, num_s, start_layer, end_layer)

        # 颜色一致性损失
        color_loss = self.calc_histogram_loss(gimage, style, self.hist)

        return gimage, loss_c, loss_s, loss_ccp, color_loss


class TestNet(nn.Module):
    def __init__(self, content_encoder, style_encoder, style_transfer, decoder):
        super(TestNet, self).__init__()
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.decoder = decoder
        self.style_transfer = style_transfer

    def forward(self, content, style):
        style_feats = self.style_encoder(style)
        content_feats = self.content_encoder(content)
        fsc = self.style_transfer(content_feats, style_feats)
        res = self.decoder(fsc)
        return res