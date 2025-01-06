import torch.nn as nn
import torch
import torch.nn.functional as F
import lib.common as common

class UNet_4layer(nn.Module):
    def __init__(self, in_dim, n_classes, depth=4, n_filters=16, drop_prob=0.1, y_range = None, kernel_size=3):
        super(UNet_4layer, self).__init__()
        self.ds_conv_1 = common.ConvBlock(in_dim, n_filters, k_size=kernel_size)
        self.ds_conv_2 = common.ConvBlock(n_filters, 2 * n_filters, k_size=kernel_size)
        self.ds_conv_3 = common.ConvBlock(2 * n_filters, 4 * n_filters, k_size=kernel_size)
        self.ds_conv_4 = common.ConvBlock(4 * n_filters, 8 * n_filters, k_size=kernel_size)

        self.ds_maxpool_1 = common.maxpool()
        self.ds_maxpool_2 = common.maxpool()
        self.ds_maxpool_3 = common.maxpool()
        self.ds_maxpool_4 = common.maxpool()

        self.ds_dropout_1 = common.dropout(drop_prob)
        self.ds_dropout_2 = common.dropout(drop_prob)
        self.ds_dropout_3 = common.dropout(drop_prob)
        self.ds_dropout_4 = common.dropout(drop_prob)

        self.bridge = common.ConvBlock(8 * n_filters, 16 * n_filters, k_size=kernel_size)

        self.us_tconv_4 = common.TransposeConvBlock(16 * n_filters, 8 * n_filters, k_size=kernel_size)
        self.us_tconv_3 = common.TransposeConvBlock(8 * n_filters, 4 * n_filters, k_size=kernel_size)
        self.us_tconv_2 = common.TransposeConvBlock(4 * n_filters, 2 * n_filters, k_size=kernel_size)
        self.us_tconv_1 = common.TransposeConvBlock(2 * n_filters, n_filters, k_size=kernel_size)

        self.us_conv_4 = common.ConvBlock(16 * n_filters, 8 * n_filters, k_size=kernel_size)
        self.us_conv_3 = common.ConvBlock(8 * n_filters, 4 * n_filters, k_size=kernel_size)
        self.us_conv_2 = common.ConvBlock(4 * n_filters, 2 * n_filters, k_size=kernel_size)
        self.us_conv_1 = common.ConvBlock(2 * n_filters, 1 * n_filters, k_size=kernel_size)

        self.us_dropout_4 = common.dropout(drop_prob)
        self.us_dropout_3 = common.dropout(drop_prob)
        self.us_dropout_2 = common.dropout(drop_prob)
        self.us_dropout_1 = common.dropout(drop_prob)

        self.output = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def forward(self, x):
        res = x

        res = self.ds_conv_1(res); conv_stack_1 = res.clone()
        res = self.ds_maxpool_1(res)
        res = self.ds_dropout_1(res)

        res = self.ds_conv_2(res); conv_stack_2 = res.clone()
        res = self.ds_maxpool_2(res)
        res = self.ds_dropout_2(res)

        res = self.ds_conv_3(res); conv_stack_3 = res.clone()
        res = self.ds_maxpool_3(res)
        res = self.ds_dropout_3(res)

        res = self.ds_conv_4(res); conv_stack_4 = res.clone()
        res = self.ds_maxpool_4(res)
        res = self.ds_dropout_4(res)

        res = self.bridge(res)

        res = self.us_tconv_4(res)
        res = torch.cat([res, conv_stack_4], dim=1)
        res = self.us_dropout_4(res)
        res = self.us_conv_4(res)

        res = self.us_tconv_3(res)
        res = torch.cat([res, conv_stack_3], dim=1)
        res = self.us_dropout_3(res)
        res = self.us_conv_3(res)

        res = self.us_tconv_2(res)
        res = torch.cat([res, conv_stack_2], dim=1)
        res = self.us_dropout_2(res)
        res = self.us_conv_2(res)

        res = self.us_tconv_1(res)
        res = torch.cat([res, conv_stack_1], dim=1)
        res = self.us_dropout_1(res)
        res = self.us_conv_1(res)

        output = self.output(res)

        return output
