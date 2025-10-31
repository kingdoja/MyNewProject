import torch
import gc
# 如果使用了GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()
# 同时建议运行垃圾回收
gc.collect()


import torch
torch.cuda.empty_cache()
# 设置内存分配策略以减少碎片
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
