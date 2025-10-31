"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw

from src.core import YAMLConfig


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        # 加载checkpoint并打印调试信息
        checkpoint = torch.load(args.resume, map_location='cpu') 
        print(f"=== 模型文件信息 ===")
        print(f"文件路径: {args.resume}")
        print(f"Checkpoint类型: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # 尝试多种可能的键名
            state = None
    
    # 优先尝试 'model' 键
            if 'model' in checkpoint and checkpoint['model'] is not None:
                state = checkpoint['model']
                print(f"✓ 使用 'model' 键")
    # 尝试 'ema' 键
            elif 'ema' in checkpoint and checkpoint['ema'] is not None:
                if isinstance(checkpoint['ema'], dict) and 'module' in checkpoint['ema']:
                    state = checkpoint['ema']['module']
                    print(f"✓ 使用 'ema.module' 键")
                else:
                    state = checkpoint['ema']
                    print(f"✓ 使用 'ema' 键")
        
    # 尝试其他可能的键
            elif 'state_dict' in checkpoint and checkpoint['state_dict'] is not None:
                state = checkpoint['state_dict']
                print(f"✓ 使用 'state_dict' 键")
            else:
        # 如果还是None，尝试第一个非None的值
                for key, value in checkpoint.items():
                    if value is not None and not key.startswith('_') and key not in ['date', 'last_epoch', 'optimizer', 'lr_scheduler', 'lr_warmup_scheduler', 'scaler']:
                        state = value
                        print(f"✓ 使用备用键: {key}")
                        break
    
            if state is None:
                print("❌ 无法找到有效的模型权重")
                print("可用的键值对:")
                for key, value in checkpoint.items():
                    print(f"  {key}: {type(value)} - {str(value)[:100]}")
                raise ValueError("模型权重加载失败")
        else:
            state = checkpoint
            print(f"✓ 直接使用checkpoint，类型: {type(checkpoint)}")

        print(f"✓ 成功加载模型权重，类型: {type(state)}")
        print(f"权重包含的键数量: {len(state) if isinstance(state, dict) else '未知'}")

# 加载到模型
        cfg.model.load_state_dict(state)
        print("✓ 模型权重加载成功")

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
