#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2021/08/21 18:10
@Author     : Zhu Shuli
@File       : RNN_Net.py
@DevTool    : PyCharm
@Desc       : RNN模型
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from tcn import TemporalConvNet


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size, use_relu=True, use_leaky=True):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.use_relu = use_relu
        if use_relu:
            if use_leaky:
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()

    def forward(self, x):
        input = x.permute(0, 2, 1)  # b*f*s -> b*s*f
        out = self.fc(input).permute(0, 2, 1)
        if self.use_relu:
            return self.relu(out)
        return out


class BasicNet(nn.Module):
    def __init__(self, acc_sizes, gy_sizes, gvt_sizes):
        super(BasicNet, self).__init__()
        # gravity embedding
        gvt_layers = []
        for i in range(1, len(gvt_sizes)):
            gvt_layers.append(FCLayer(gvt_sizes[i - 1], gvt_sizes[i]))
        self.gvt_ebd = nn.Sequential(*gvt_layers)
        # acc embedding
        acc_layers = []
        for i in range(1, len(acc_sizes)):
            acc_layers.append(FCLayer(acc_sizes[i - 1], acc_sizes[i]))
        self.acc_ebd = nn.Sequential(*acc_layers)
        # gy embedding
        gy_layers = []
        for i in range(1, len(gy_sizes)):
            gy_layers.append(FCLayer(gy_sizes[i - 1], gy_sizes[i]))
        self.gy_ebd = nn.Sequential(*gy_layers)

    def forward(self, acc, gy, gvt):
        gvt_ebd = self.gvt_ebd(torch.cat([acc, gy], dim=1)) + gvt
        acc_ebd = self.acc_ebd(torch.cat([acc, gvt_ebd], dim=1))
        gy_ebd = self.gy_ebd(torch.cat([gy, gvt_ebd], dim=1))
        return torch.cat([acc_ebd, gy_ebd], dim=1)


class BasicNetV2(nn.Module):
    def __init__(self, acc_sizes, gy_sizes):
        super(BasicNetV2, self).__init__()
        # gamerv embedding
        # grv_layers = []
        # for i in range(1, len(grv_sizes)):
        #     grv_layers.append(FCLayer(grv_sizes[i - 1], grv_sizes[i]))
        # self.grv_ebd = nn.Sequential(*grv_layers)
        # acc embedding
        acc_layers = []
        for i in range(1, len(acc_sizes)):
            acc_layers.append(FCLayer(acc_sizes[i - 1], acc_sizes[i]))
        self.acc_ebd = nn.Sequential(*acc_layers)
        # gy embedding
        gy_layers = []
        for i in range(1, len(gy_sizes)):
            gy_layers.append(FCLayer(gy_sizes[i - 1], gy_sizes[i]))
        self.gy_ebd = nn.Sequential(*gy_layers)

    def forward(self, acc, gy):
        # if gamerv is not None:
        #     gamerv_ebd = self.grv_ebd(torch.cat([acc, gy, gamerv], dim=1))
        # else:
        #     gamerv_ebd = self.grv_ebd(torch.cat([acc, gy], dim=1))
        acc_ebd = self.acc_ebd(torch.cat([acc], dim=1))
        gy_ebd = self.gy_ebd(torch.cat([gy], dim=1))
        return torch.cat([acc_ebd, gy_ebd], dim=1)
        # return torch.cat([acc, gy, gamerv], dim=1)


class PdrLSTM(nn.Module):
    def __init__(self, acc_sizes, gy_sizes, gvt_sizes, lstm_input_size, lstm_hidden_size, lstm_num_layers,
                 fc_spd_sizes, fc_brng_sizes):
        super(PdrLSTM, self).__init__()
        self.basic_net = BasicNet(acc_sizes, gy_sizes, gvt_sizes)
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        # speed fully connected
        spd_layers = []
        for i in range(1, len(fc_spd_sizes)):
            if i <= 2:
                spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=True, use_leaky=False))
            else:
                spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=False, use_leaky=False))
        self.fc_spd_out = nn.Sequential(*spd_layers)
        # bearing fully connected
        brng_layers = []
        for i in range(1, len(fc_brng_sizes)):
            if i <= 2:
                brng_layers.append(FCLayer(fc_brng_sizes[i - 1], fc_brng_sizes[i], use_relu=True, use_leaky=False))
            else:
                brng_layers.append(FCLayer(fc_brng_sizes[i - 1], fc_brng_sizes[i], use_relu=False, use_leaky=False))
        self.fc_brng_out = nn.Sequential(*brng_layers)

    def forward(self, acc, gy, gvt, init_spd=None, init_brng=None):
        basic_res = self.basic_net(acc, gy, gvt)  # b*f*s
        lstm_res = self.lstm(basic_res.permute(2, 0, 1))[0].permute(1, 2, 0)  # s*b*f -> b*f*s
        if init_spd is not None:
            spd_out = self.fc_spd_out(torch.cat([lstm_res, init_spd], dim=1))
        else:
            spd_out = self.fc_spd_out(lstm_res)
        if init_brng is not None:
            brng_out = self.fc_brng_out(torch.cat([lstm_res, init_brng], dim=1))
        else:
            brng_out = self.fc_brng_out(lstm_res)
        return torch.cat([spd_out, brng_out], dim=1)


class PdrSpdLSTM(nn.Module):
    def __init__(self, acc_sizes, gy_sizes, gvt_sizes,
                 lstm_input_size, lstm_hidden_size, lstm_num_layers, fc_spd_sizes):
        super(PdrSpdLSTM, self).__init__()
        self.basic_net = BasicNet(acc_sizes, gy_sizes, gvt_sizes)
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        # speed fully connected
        spd_layers = []
        for i in range(1, len(fc_spd_sizes)):
            if i <= 2:
                spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=True, use_leaky=False))
            else:
                spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=False, use_leaky=False))
        self.fc_spd_out = nn.Sequential(*spd_layers)

    def forward(self, acc, gy, gvt, init_spd=None, init_brng=None):
        basic_res = self.basic_net(acc, gy, gvt)  # b*f*s
        lstm_res = self.lstm(basic_res.permute(2, 0, 1))[0].permute(1, 2, 0)  # s*b*f -> b*f*s
        if init_spd is not None:
            spd_out = self.fc_spd_out(torch.cat([lstm_res, init_spd], dim=1))
        else:
            spd_out = self.fc_spd_out(lstm_res)
        return spd_out


class PdrSpdLSTMV2(nn.Module):
    def __init__(self, acc_sizes, gy_sizes,
                 lstm_input_size, lstm_hidden_size, lstm_num_layers, fc_spd_sizes):
        super(PdrSpdLSTMV2, self).__init__()
        self.basic_net = BasicNetV2(acc_sizes, gy_sizes)
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        # speed fully connected
        spd_layers = []
        for i in range(1, len(fc_spd_sizes)):
            if i <= 2:
                spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=True, use_leaky=False))
            else:
                spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=False, use_leaky=False))
        self.fc_spd_out = nn.Sequential(*spd_layers)

    def forward(self, acc, gy, init_spd=None, pose=None, src_mask=None):
        basic_res = self.basic_net(acc, gy)  # b*f*s
        lstm_res = self.lstm(basic_res.permute(2, 0, 1))[0].permute(1, 2, 0)  # s*b*f -> b*f*s
        if init_spd is not None and pose is not None:
            spd_out = self.fc_spd_out(torch.cat([lstm_res, init_spd, pose], dim=1))
        elif init_spd is not None:
            spd_out = self.fc_spd_out(torch.cat([lstm_res, init_spd], dim=1))
        else:
            spd_out = self.fc_spd_out(lstm_res)
        return spd_out


# class PdrSpdTCN(nn.Module):
#     def __init__(self, acc_sizes, gy_sizes, gvt_sizes, tcn_input, tcn_channels, tcn_ks, fc_spd_sizes, tcn_dropout=0.5):
#         super(PdrSpdTCN, self).__init__()
#         self.basic_net = BasicNet(acc_sizes, gy_sizes, gvt_sizes)
#         self.tcn = TemporalConvNet(tcn_input, tcn_channels, kernel_size=tcn_ks, dropout=tcn_dropout)
#         # speed fully connected
#         spd_layers = []
#         for i in range(1, len(fc_spd_sizes)):
#             if i <= 2:
#                 spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=True, use_leaky=False))
#             else:
#                 spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=False, use_leaky=False))
#         self.fc_spd_out = nn.Sequential(*spd_layers)
#
#     def forward(self, acc, gy, gvt, init_spd=None):
#         basic_res = self.basic_net(acc, gy, gvt)  # b*f*s
#         tcn_res = self.tcn(basic_res)
#         if init_spd is not None:
#             spd_out = self.fc_spd_out(torch.cat([tcn_res, init_spd], dim=1))
#         else:
#             spd_out = self.fc_spd_out(tcn_res)
#         return spd_out[:, :, ::50]
#
#
# class PdrSpdTCNV2(nn.Module):
#     def __init__(self, acc_sizes, gy_sizes, gamerv_sizes,
#                  tcn_input, tcn_channels, tcn_ks, fc_spd_sizes, tcn_dropout=0.5):
#         super(PdrSpdTCNV2, self).__init__()
#         self.basic_net = BasicNetV2(acc_sizes, gy_sizes, gamerv_sizes)
#         self.tcn = TemporalConvNet(tcn_input, tcn_channels, kernel_size=tcn_ks, dropout=tcn_dropout)
#         # speed fully connected
#         spd_layers = []
#         for i in range(1, len(fc_spd_sizes)):
#             if i <= 2:
#                 spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=True, use_leaky=False))
#             else:
#                 spd_layers.append(FCLayer(fc_spd_sizes[i - 1], fc_spd_sizes[i], use_relu=False, use_leaky=False))
#         self.fc_spd_out = nn.Sequential(*spd_layers)
#
#     def forward(self, acc, gy, gvt, init_spd=None):
#         basic_res = self.basic_net(acc, gy, gvt)  # b*f*s
#         tcn_res = self.tcn(basic_res)
#         if init_spd is not None:
#             spd_out = self.fc_spd_out(torch.cat([tcn_res, init_spd], dim=1))
#         else:
#             spd_out = self.fc_spd_out(tcn_res)
#         return spd_out[:, :, ::50]


def load_rnn_net(version=None, pretrained_model=None, trained_model=None):
    if version is None:
        model = PdrSpdLSTMV2(acc_sizes=(750, 256, 128, 64, 32), gy_sizes=(750, 256, 128, 64, 32),
                             lstm_input_size=64, lstm_hidden_size=128, lstm_num_layers=1,
                             fc_spd_sizes=(128 + 1, 64, 32, 16, 1))
    elif version == 'v1.0':
        model = PdrSpdLSTMV2(acc_sizes=(12 + 128, 64, 32), gy_sizes=(12 + 128, 64, 32),
                             grv_sizes=(12 + 12, 64, 128),
                             lstm_input_size=64, lstm_hidden_size=128, lstm_num_layers=1,
                             fc_spd_sizes=(128 + 1, 64, 32, 16, 1))
    elif version == 'v2.0':
        model = PdrSpdLSTMV2(acc_sizes=(30 + 128, 64, 32), gy_sizes=(30 + 128, 64, 32),
                             grv_sizes=(30 + 30, 64, 128),
                             lstm_input_size=64, lstm_hidden_size=128, lstm_num_layers=1,
                             fc_spd_sizes=(128 + 1, 64, 32, 16, 1))
    else:
        raise ValueError('Feature Version is Illegal.')

    if pretrained_model is not None and trained_model is None:
        print('Load pretrained model: %s' % pretrained_model)
        model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    if trained_model is not None:
        print('Load trained model: %s' % trained_model)
        model.load_state_dict(torch.load(trained_model, map_location='cpu'))
    return model


__all__ = ['load_rnn_net', 'PdrSpdLSTMV2', 'BasicNetV2', 'FCLayer']

if __name__ == '__main__':
    # tcn = TemporalConvNet(3, (32, 64), kernel_size=5, dropout=0.5)
    # print(tcn.__str__())

    # pdr_spd_net = PdrSpdTCN(acc_sizes=(64 + 3, 64, 128), gy_sizes=(64 + 3, 64, 128), gvt_sizes=(3, 32, 64),
    #                         tcn_input=256, tcn_channels=(256, 128, 128), tcn_ks=5,
    #                         fc_spd_sizes=(128 + 1, 64, 32, 16, 1))
    # print(pdr_spd_net.__str__())

    model = load_rnn_net(version='v1.0')
    print(model.__str__())
