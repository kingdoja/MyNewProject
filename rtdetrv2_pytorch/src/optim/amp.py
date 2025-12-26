"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch

from ..core import register


__all__ = ['GradScaler']

# 使用新的 API: torch.amp.GradScaler('cuda', ...) 替代 torch.cuda.amp.grad_scaler.GradScaler
# 为了兼容性，检查 PyTorch 版本
if hasattr(torch.amp, 'GradScaler'):
    # PyTorch >= 2.0 使用新 API（不会触发警告）
    @register()
    class GradScaler:
        """包装 torch.amp.GradScaler 以使用新的 API"""
        def __init__(self, *args, **kwargs):
            # 使用新的 API: torch.amp.GradScaler('cuda', ...)
            self._scaler = torch.amp.GradScaler('cuda', *args, **kwargs)
        
        def __getattr__(self, name):
            # 委托所有属性访问到内部的 scaler
            return getattr(self._scaler, name)
else:
    # 旧版本回退到旧 API
    # 注意：在旧版本 PyTorch 中，这个 API 还没有被弃用，所以不会有警告
    import torch.cuda.amp as amp
    GradScaler = register()(amp.grad_scaler.GradScaler)
