import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spatial_correlation_sampler import SpatialCorrelationSampler
from .Resnet18 import *

class MaskPropagation(nn.Module):
    def __init__(self, downsample_rate=4, search_radius=6, max_obj_num=16):
        super(MaskPropagation, self).__init__()
        self.D = downsample_rate
        self.R = search_radius  # window size
        self.C = max_obj_num
        self.P = self.R * 2 + 1

        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.P,
            stride=1,
            padding=0,
            dilation=1)

    def downsample_msk(self, image):
        x = image.float()[:, :, ::self.D, ::self.D]
        if not self.training and image.size()[1] == 1:
            x = mask_to_one_hot(x.long(), self.C)
        return x

    def forward(self, feat_mem, feat_query, msk_mem, feat_flo, optical_flows=None):
        if self.training:
            b, c, h, w = feat_query.size()
            corrs = []
            for ind in range(len(feat_mem)):
                corrs.append(self.correlation_sampler(feat_query, feat_mem[ind]))
                _, _, _, h1, w1 = corrs[-1].size()
                corrs[ind] = corrs[ind].reshape([b, self.P * self.P, h1 * w1])
            corr = torch.cat(corrs, 1) / torch.sqrt(torch.tensor(c).float())
            corr = F.softmax(corr, dim=1).unsqueeze(1)
            feat_flo = feat_flo.expand(b, self.P * self.P, h1, w1)
            feat_flo = feat_flo.reshape(b, self.P * self.P, h1 * w1)
            feat_option = feat_flo / torch.sqrt(torch.tensor(c).float())
            feat_option = feat_option.unsqueeze(1)
            corr = corr + feat_option
            msk_mem = [self.downsample_msk(msk) for msk in msk_mem]
            obj_num = msk_mem[0].size(1)
            msk_mem = [F.unfold(msk, kernel_size=self.P, padding=self.R) for msk in msk_mem]
            msk_mem = [msk.reshape([b, obj_num, self.P * self.P, h * w]) for msk in msk_mem]
            msk_mem = torch.cat(msk_mem, 2)
            out = (corr * msk_mem).sum(2).reshape([b, obj_num, h, w])
        else:
            b, c, h, w = feat_query.size()
            corrs = []
            for ind in range(len(feat_mem)):
                corrs.append(self.correlation_sampler(feat_query, feat_mem[ind]))
                _, _, _, h1, w1 = corrs[-1].size()
                corrs[ind] = corrs[ind].reshape([b, self.P * self.P, h1 * w1])
            corr = torch.cat(corrs, 1) / torch.sqrt(torch.tensor(c).float())
            corr = corr.unsqueeze(1)
            corr, top_ind = torch.topk(corr, 36, dim=2)
            corr = F.softmax(corr, dim=2)
            feat_flo = feat_flo.expand(len(feat_mem), self.P * self.P, h1, w1)
            feat_flo = feat_flo.reshape(len(feat_mem), self.P * self.P, h1 * w1)
            k = feat_flo.chunk(len(feat_mem), dim=0)
            feat_flo = torch.cat(k, 1)
            feat_option = feat_flo / torch.sqrt(torch.tensor(c).float())
            feat_option = feat_option.unsqueeze(1)
            feat_option, top_ind1 = torch.topk(feat_option, 36, dim=2)
            corr = corr + feat_option
            msk_mem = [self.downsample_msk(msk) for msk in msk_mem]
            obj_num = msk_mem[0].size(1)
            msk_mem = [F.unfold(msk, kernel_size=self.P, padding=self.R) for msk in msk_mem]
            msk_mem = [msk.reshape([b, obj_num, self.P * self.P, h * w]) for msk in msk_mem]
            msk_mem = torch.cat(msk_mem, 2)
            top_ind = top_ind.expand(-1, obj_num, -1, -1)
            msk_mem = torch.gather(msk_mem, dim=2, index=top_ind)
            out = (corr * msk_mem).sum(2).reshape([b, obj_num, h, w])
        return out

class MFC(nn.Module):
    def __init__(self, args):
        super(MFC, self).__init__()
        self.args = args
        self.max_obj_num = 7  # >= max object number in each video of DAVIS and YOUTUBE
        self.feature_extraction = ResNet18(in_ch=3)
        self.feature_extraction1 = ResNet18(in_ch=2)
        self.post_convolution = nn.Conv2d(256, 64, 3, 1, 1)
        self.post_convolution1 = nn.Conv2d(256, 1, 3, 1, 1)
        self.downsample_rate = 4  # according to ResNet18
        if args.training:
            self.search_radius = args.train_corr_radius
        else:
            self.search_radius = args.test_corr_radius
        self.propagation = MaskPropagation(self.downsample_rate, self.search_radius, self.max_obj_num)

    def forward(self, img_mem, msk_mem, img_query, img_option, optical_flows=None):
        with autocast(enabled=self.args.is_amp):
            feat_mem = [self.post_convolution(self.feature_extraction(img)) for img in img_mem]
            feat_query = self.post_convolution(self.feature_extraction(img_query))
            feat_flo = self.post_convolution1(self.feature_extraction1(img_option))
        results = self.propagation(feat_mem, feat_query, msk_mem, feat_flo, optical_flows)
        return results
    def dropout2d_lab(self, arr):
        if not self.training:
            return arr
        drop_ch_ind = np.random.choice(np.arange(1, 3), 1, replace=False)
        for a in arr:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= 3 / 2
        return arr, drop_ch_ind

class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.conv_query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.conv_key(x).view(batch_size, -1, height * width)
        value = self.conv_value(x).view(batch_size, -1, height * width)
        attention_map = torch.bmm(query, key)
        attention_map = F.softmax(attention_map, dim=-1)
        feature_map = torch.bmm(value, attention_map.permute(0, 2, 1))
        feature_map = feature_map.view(batch_size, channels, height, width)
        feature_map = self.gamma * feature_map + x
        return feature_map
