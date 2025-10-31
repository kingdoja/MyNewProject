"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 


class AdaptiveFeatureEnhancement(nn.Module):
    """
    自适应特征增强模块 (AFE)
    通过通道注意力和空间注意力增强特征表达
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(AdaptiveFeatureEnhancement, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # 空间注意力
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights
        
        return x


class CrossLevelAttentionFusion(nn.Module):
    """
    跨层注意力融合机制 (CLAF)
    融合不同层级的特征以增强表达能力
    """
    def __init__(self, channels):
        super(CrossLevelAttentionFusion, self).__init__()
        self.channels = channels
        self.fusion_conv = nn.ModuleList([
            nn.Conv2d(channels[i], channels[0], kernel_size=1) 
            for i in range(len(channels))
        ])
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[0] * len(channels), channels[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], len(channels), kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        # 将所有特征调整到同一尺寸（以最低分辨率为准）
        target_size = features[-1].shape[2:]
        upsampled_features = []
        
        for i, feat in enumerate(features):
            # 降通道
            feat = self.fusion_conv[i](feat)
            # 上采样到目标尺寸
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat)
        
        # 拼接所有特征
        concat_features = torch.cat(upsampled_features, dim=1)
        
        # 计算注意力权重
        attention_weights = self.attention(concat_features)
        
        # 应用注意力权重
        weighted_features = []
        for i in range(len(features)):
            weight = attention_weights[:, i:i+1]
            weighted_features.append(upsampled_features[i] * weight)
        
        # 融合特征
        fused_feature = sum(weighted_features)
        
        return fused_feature


class RTDETRImproved(nn.Module):
    """
    改进的RT-DETR模型，包含AFE和CLAF模块
    """
    __inject__ = [
        'backbone',
        'encoder',
        'decoder',
    ]

    def __init__(self,
                 backbone: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 afe_config=None,
                 claf_config=None,
                 dla_enabled=False):
        
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.dla_enabled = dla_enabled
        
        # 初始化AFE模块
        if afe_config:
            self.afe_modules = nn.ModuleList([
                AdaptiveFeatureEnhancement(channels, afe_config.reduction_ratio)
                for channels in afe_config.in_channels
            ])
        else:
            self.afe_modules = None
            
        # 初始化CLAF模块
        if claf_config:
            self.claf_module = CrossLevelAttentionFusion(claf_config.channels)
        else:
            self.claf_module = None

    def forward(self, x, targets=None):
        # Backbone特征提取
        feats = self.backbone(x)
        
        # 应用自适应特征增强模块
        if self.afe_modules:
            enhanced_feats = []
            for i, feat in enumerate(feats):
                enhanced_feat = self.afe_modules[i](feat)
                enhanced_feats.append(enhanced_feat)
            feats = enhanced_feats
        
        # 应用跨层注意力融合机制
        if self.claf_module:
            fused_feature = self.claf_module(feats)
            # 将融合后的特征替换最后一层特征
            feats[-1] = fused_feature
        
        # Encoder处理
        feats = self.encoder(feats)
        
        # Decoder处理
        out = self.decoder(feats, targets)
        
        return out
