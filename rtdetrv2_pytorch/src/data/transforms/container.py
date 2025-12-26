""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T

from typing import Any, Dict, List, Optional

from ._transforms import EmptyTransform, RandomTransformWithP
from ...core import register, GLOBAL_CONFIG


@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    # 检查是否有 p 参数，以及 transform 是否已经支持 p 参数
                    p = op.pop('p', None)
                    # 如果 transform 是 RandomIoUCrop 且指定了 p，需要将 p 参数传给它
                    if name == 'RandomIoUCrop' and p is not None:
                        op['p'] = p
                    # 创建 transform 实例
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], GLOBAL_CONFIG[name]['_name'])(**op)
                    # 如果指定了 p 参数且 transform 不支持 p（不是 RandomIoUCrop），则用 RandomTransformWithP 包装
                    if p is not None and name != 'RandomIoUCrop':
                        transfom = RandomTransformWithP(transform=transfom, p=p)
                    transforms.append(transfom)
                    # 恢复 op 字典（用于调试或保持原始配置）
                    op['type'] = name
                    if p is not None:
                        op['p'] = p

                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]
 
        super().__init__(transforms=transforms)

        if policy is None:
            policy = {'name': 'default'}
        elif isinstance(policy, dict) and 'name' not in policy:
            # 如果 policy 是字典但没有 'name' 键，添加默认值
            policy = {'name': 'default', **policy}

        self.policy = policy
        self.global_samples = 0

    def forward(self, *inputs: Any) -> Any:
        policy_name = self.policy.get('name', 'default')
        return self.get_forward(policy_name)(*inputs)

    def get_forward(self, name):
        forwards = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
            'stop_sample': self.stop_sample_forward,
        }
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        
        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_epoch = self.policy['epoch']

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                pass
            else:
                sample = transform(sample)

        return sample


    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        
        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_sample = self.policy['sample']

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample
