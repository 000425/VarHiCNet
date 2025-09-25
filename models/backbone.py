# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, feature_fusion=None, aspp=None):
        super().__init__(backbone, position_embedding)
        self.feature_fusion = feature_fusion
        self.aspp = aspp
        
        # 设置输出通道数
        if feature_fusion is not None:
            self.num_channels = 256
        elif aspp is not None:
            self.num_channels = 256  # ASPP输出通道数
        else:
            self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)  # Backbone输出
        out: List[NestedTensor] = []
        pos = []
        
        # 处理所有层特征
        for name, x in sorted(xs.items()):
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))  # 位置编码
        
        # 特征融合（如果启用）
        if self.feature_fusion is not None:
            # 融合最后三层(layer2,3,4)
            fused = self.feature_fusion(out[1:4])
            # 应用ASPP（如果启用）
            if self.aspp is not None:
                fused = self.aspp(fused)
            return [fused], [pos[3]]  # 返回融合特征和layer4的位置编码
        
        # 单独应用ASPP（如果没有特征融合）
        elif self.aspp is not None:
            # 只对最后一层应用ASPP
            last_layer = out[-1]
            aspp_out = self.aspp(last_layer)
            return [aspp_out], [pos[-1]]
        
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = True  # 始终返回多层特征以支持融合
    
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    
    # 配置特征融合
    feature_fusion = None
    if args.use_feature_fusion:
        if args.backbone in ['resnet18', 'resnet34']:
            in_channels_list = [128, 256, 512]
        else:  # resnet50/101
            in_channels_list = [512, 1024, 2048]
        feature_fusion = FeatureFusionModule(in_channels_list)
    
    # 配置ASPP - 输入通道固定为256（特征融合输出通道）
    aspp = None
    if args.use_aspp:
        # 无论backbone类型如何，ASPP输入通道始终为256
        aspp = ASPP(in_channels=256, out_channels=256)
    
    model = Joiner(backbone, position_embedding, feature_fusion, aspp)
    return model



class FeatureFusionModule(nn.Module):
    """
    带有注意力机制的特征融合模块
    融合ResNet的layer2, layer3, layer4特征
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        # 调整各层通道数
        self.conv2 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.conv4 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        
        # 注意力权重生成
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 3, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # 初始化参数
        for conv in [self.conv2, self.conv3, self.conv4]:
            nn.init.kaiming_uniform_(conv.weight, a=1)
            nn.init.constant_(conv.bias, 0)
        for m in self.attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: List[NestedTensor]):
        # 获取各层特征和掩码
        f2_tensor = features[0].tensors
        f3_tensor = features[1].tensors
        f4_tensor = features[2].tensors
        mask = features[2].mask  # 使用layer4的掩码
        
        # 调整通道数
        f2 = self.conv2(f2_tensor)
        f3 = self.conv3(f3_tensor)
        f4 = self.conv4(f4_tensor)
        
        # 上采样到最大尺寸(layer4)
        f2 = F.interpolate(f2, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        
        # 拼接特征并生成注意力权重
        combined = torch.cat([f2, f3, f4], dim=1)
        weights = self.attention(combined)  # [B, 3, H, W]
        
        # 拆分权重并加权融合
        w2, w3, w4 = weights.chunk(3, dim=1)
        fused = f2 * w2 + f3 * w3 + f4 * w4
        
        # 返回与layer4尺寸相同的融合特征
        return NestedTensor(fused, mask)
    

class ASPP(nn.Module):
    """ASPP模块用于捕获多尺度上下文信息"""
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super().__init__()
        modules = []
        # 1x1卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        # 多个不同空洞率的空洞卷积
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, 
                         dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        
        # 全局平均池化分支
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        self.convs = nn.ModuleList(modules)
        
        # 项目卷积，融合所有分支
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: NestedTensor):
        # 分解NestedTensor
        tensor = x.tensors
        mask = x.mask
        
        # 存储各个分支的结果
        res = []
        for conv in self.convs:
            # 处理全局池化分支（需要上采样）
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                pool_feat = conv(tensor)
                res.append(F.interpolate(pool_feat, size=tensor.shape[-2:], 
                                        mode='bilinear', align_corners=False))
            else:
                res.append(conv(tensor))
        
        # 拼接所有分支结果
        res = torch.cat(res, dim=1)
        
        # 融合特征
        res = self.project(res)
        return NestedTensor(res, mask)