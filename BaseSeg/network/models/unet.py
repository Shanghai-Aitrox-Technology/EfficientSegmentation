
import torch
import torch.nn as nn

from BaseSeg.network.blocks.basic_unit import _ConvINReLU3D
from BaseSeg.network.blocks.process_block import InputLayer, OutputLayer
from BaseSeg.network.blocks.residual_block import ResTwoLayerConvBlock, ResFourLayerConvBlock


class UNet(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()

        # UNet parameter.
        num_class = cfg['NUM_CLASSES']
        num_channel = cfg['NUM_CHANNELS']
        self.num_depth = cfg['NUM_DEPTH']
        self.is_preprocess = cfg['IS_PREPROCESS']
        self.is_postprocess = cfg['IS_POSTPROCESS']
        self.auxiliary_task = cfg['AUXILIARY_TASK']
        self.auxiliary_class = cfg['AUXILIARY_CLASS']
        self.is_dynamic_empty_cache = cfg['IS_DYNAMIC_EMPTY_CACHE']

        if cfg['ENCODER_CONV_BLOCK'] == 'ResTwoLayerConvBlock':
            encoder_conv_block = ResTwoLayerConvBlock
        else:
            encoder_conv_block = ResFourLayerConvBlock
        if cfg['DECODER_CONV_BLOCK'] == 'ResTwoLayerConvBlock':
            decoder_conv_block = ResTwoLayerConvBlock
        else:
            decoder_conv_block = ResFourLayerConvBlock

        self.input = InputLayer(input_size=cfg['INPUT_SIZE'], clip_window=cfg['WINDOW_LEVEL'])
        self.output = OutputLayer()

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = encoder_conv_block(1, num_channel[0], num_channel[0],
                                          is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        self.conv1_0 = encoder_conv_block(num_channel[0], num_channel[1], num_channel[1],
                                          is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        self.conv2_0 = encoder_conv_block(num_channel[1], num_channel[2], num_channel[2],
                                          is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        self.conv3_0 = encoder_conv_block(num_channel[2], num_channel[3], num_channel[3],
                                          is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        self.conv4_0 = encoder_conv_block(num_channel[3], num_channel[4], num_channel[4],
                                          is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        if self.num_depth == 5:
            self.conv5_0 = encoder_conv_block(num_channel[4], num_channel[5], num_channel[5],
                                              is_dynamic_empty_cache=self.is_dynamic_empty_cache)
            self.conv4_1 = decoder_conv_block(num_channel[4] + num_channel[5], num_channel[4], num_channel[4],
                                              is_dynamic_empty_cache=self.is_dynamic_empty_cache)

        self.conv3_1 = decoder_conv_block(num_channel[3] + num_channel[4], num_channel[3], num_channel[3],
                                          is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        self.conv2_2 = decoder_conv_block(num_channel[2] + num_channel[3], num_channel[2], num_channel[2],
                                          is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        self.conv1_3 = decoder_conv_block(num_channel[1] + num_channel[2], num_channel[1], num_channel[1],
                                          is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        self.conv0_4 = decoder_conv_block(num_channel[0] + num_channel[1], num_channel[0], num_channel[0],
                                          is_dynamic_empty_cache=self.is_dynamic_empty_cache)

        self.final = nn.Conv3d(num_channel[0], num_class, kernel_size=1, bias=False)
        if self.auxiliary_task:
            self.final1 = nn.Sequential(_ConvINReLU3D(num_channel[2], num_channel[2], kernel_size=3, padding=1, p=0.2),
                                        nn.Conv3d(num_channel[2], self.auxiliary_class, kernel_size=1, bias=False))

        self._initialize_weights()
        # self.final.bias.data.fill_(-2.19)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out_size = x.shape[2:]
        if self.is_preprocess:
            x = self.input(x)

        x = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        if self.num_depth == 5:
            x5_0 = self.conv5_0(self.pool(x4_0))
            x4_0 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
            if self.is_dynamic_empty_cache:
                del x5_0
                torch.cuda.empty_cache()

        x3_0 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        if self.is_dynamic_empty_cache:
            del x4_0
            torch.cuda.empty_cache()

        x2_0 = self.conv2_2(torch.cat([x2_0, self.up(x3_0)], 1))
        if self.is_dynamic_empty_cache:
            del x3_0
            torch.cuda.empty_cache()
        if self.auxiliary_task:
            out_1 = self.final1(x2_0)

        x1_0 = self.conv1_3(torch.cat([x1_0, self.up(x2_0)], 1))
        if self.is_dynamic_empty_cache:
            del x2_0
            torch.cuda.empty_cache()

        x = self.conv0_4(torch.cat([x, self.up(x1_0)], 1))
        if self.is_dynamic_empty_cache:
            del x1_0
            torch.cuda.empty_cache()

        x = self.final(x)
        if self.is_postprocess:
            x = self.output(x, out_size)

        if self.auxiliary_task:
            return [x, out_1]
        else:
            return x
