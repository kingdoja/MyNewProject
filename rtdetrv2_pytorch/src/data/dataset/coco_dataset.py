"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
from faster_coco_eval.utils.pytorch import FasterCocoDetection
import torchvision

from PIL import Image 
from faster_coco_eval.core import mask as coco_mask

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

__all__ = ['CocoDetection']

torchvision.disable_beta_transforms_warning()

@register()
class CocoDetection(FasterCocoDetection, DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False):
        # Use standard MRO call so FasterCocoDetection initializes `coco` and `ids`.
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.remap_mscoco_category = remap_mscoco_category
        if not hasattr(self, 'ids') or not hasattr(self, 'coco'):
            raise AttributeError(
                "CocoDetection init failed: missing `ids` or `coco` from FasterCocoDetection."
            )

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            # 对自定义数据集，将原始 category_id 映射为 [0, num_classes-1] 的连续索引
            image, target = self.prepare(image, target, category2label=self.category2label)

        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')
        
        # === 新增，将原图高宽保存到 target['orig_size']，顺序为(高,宽) ===
        if isinstance(image, Image.Image):
            width, height = image.size  # PIL为(w, h)
        elif torch.is_tensor(image):
            # 假设[T, H, W]或者[H, W, C]
            shape = image.shape
            height, width = shape[-2], shape[-1]
        else:
            height = width = None # fallback
        if (height is not None) and (width is not None):
            target['orig_size'] = torch.tensor([height, width])
        
        return image, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s 

    @property
    def categories(self, ):
        return self.coco.dataset['categories']

    @property
    def category2name(self, ):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self, ):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self, ):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}


def compute_bbox_from_polygon(segmentation):
    """从多边形segmentation计算边界框(bbox)。
    
    支持格式：
    - 单个多边形：list[float]，格式为 [x1, y1, x2, y2, ...]
    - 多个多边形：list[list[float]]
    - RLE格式：dict，包含 'counts'，返回None（需要从mask计算）
    
    返回：
    - bbox: [x_min, y_min, width, height] 格式，如果无法计算则返回None
    """
    if segmentation is None or (isinstance(segmentation, list) and len(segmentation) == 0):
        return None
    
    # RLE格式，无法直接计算bbox
    if isinstance(segmentation, dict) and 'counts' in segmentation:
        return None
    
    # 多边形格式
    if isinstance(segmentation, list):
        all_points = []
        
        # 检查是否是多个多边形的列表
        if len(segmentation) > 0 and isinstance(segmentation[0], list):
            # 多个多边形情况
            for poly in segmentation:
                if isinstance(poly, (list, tuple)) and len(poly) >= 6:  # 至少3个点
                    # 提取所有点坐标 [x1, y1, x2, y2, ...]
                    for i in range(0, len(poly), 2):
                        if i + 1 < len(poly):
                            all_points.append((poly[i], poly[i+1]))
        else:
            # 单个多边形情况 [x1, y1, x2, y2, ...]
            if len(segmentation) >= 6:  # 至少3个点
                for i in range(0, len(segmentation), 2):
                    if i + 1 < len(segmentation):
                        all_points.append((segmentation[i], segmentation[i+1]))
        
        if len(all_points) > 0:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            width = x_max - x_min
            height = y_max - y_min
            # 返回COCO格式的bbox: [x_min, y_min, width, height]
            return [x_min, y_min, width, height]
    
    return None


def convert_coco_poly_to_mask(segmentations, height, width):
    """将COCO分割注释转为二值mask，鲁棒处理空/无效segmentation。

    支持格式：
    - 多边形：list[list[float]]；自动过滤点数少于3个点(长度<6)的多边形
    - RLE：dict，包含 'counts'
    - 为空或无效时跳过
    """
    masks = []
    for seg in segmentations:
        try:
            rles = None
            # RLE dict
            if isinstance(seg, dict) and 'counts' in seg:
                rles = seg
            # 多边形 list
            elif isinstance(seg, list):
                # 过滤无效多边形
                valid_polys = [p for p in seg if isinstance(p, (list, tuple)) and len(p) >= 6]
                if len(valid_polys) == 0:
                    # 如果没有有效多边形，创建一个空的mask
                    mask = torch.zeros((height, width), dtype=torch.uint8)
                    masks.append(mask)
                    continue
                rles = coco_mask.frPyObjects(valid_polys, height, width)
            else:
                # 其他未知格式，创建空mask
                mask = torch.zeros((height, width), dtype=torch.uint8)
                masks.append(mask)
                continue

            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            masks.append(mask)
        except Exception:
            # 任意单条失败时跳过该实例，不中断整个batch
            # 创建一个空的mask作为占位符
            mask = torch.zeros((height, width), dtype=torch.uint8)
            masks.append(mask)
            continue

    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # 处理bbox：如果标注有segmentation但没有bbox，则从segmentation计算bbox
        # 同时过滤掉无法获取bbox的标注，确保anno和boxes保持同步
        boxes = []
        valid_anno = []
        for obj in anno:
            bbox = None
            if "bbox" in obj and obj["bbox"] is not None:
                # 如果已有bbox，直接使用
                bbox = obj["bbox"]
            elif "segmentation" in obj and obj["segmentation"] is not None:
                # 如果只有segmentation，从多边形计算bbox
                computed_bbox = compute_bbox_from_polygon(obj["segmentation"])
                if computed_bbox is not None:
                    bbox = computed_bbox
                    # 同时更新原对象中的bbox字段，以便后续使用
                    obj["bbox"] = computed_bbox
                    # 如果没有area字段，也计算一下
                    if "area" not in obj or obj["area"] is None:
                        obj["area"] = computed_bbox[2] * computed_bbox[3]
            
            # 只有成功获取到bbox的标注才保留
            if bbox is not None:
                boxes.append(bbox)
                valid_anno.append(obj)
        
        # 使用过滤后的anno列表
        anno = valid_anno
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]
            
        labels = torch.tensor(labels, dtype=torch.int64)

        masks = None
        masks_aligned = False
        if self.return_masks:
            segmentations = []
            for obj in anno:
                seg = obj.get("segmentation", None)
                # 允许为空，统一交给转换函数处理
                if seg is None:
                    seg = []
                segmentations.append(seg)
            masks_tmp = convert_coco_poly_to_mask(segmentations, h, w)
            # 只有当生成的mask数量与标注数量一致时，才认为可与boxes对齐
            if masks_tmp.shape[0] == len(anno):
                masks = masks_tmp
                masks_aligned = True

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks and masks_aligned and masks is not None and masks.shape[0] == keep.shape[0]:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks and masks_aligned and masks is not None and masks.shape[0] == boxes.shape[0]:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno], dtype=torch.float32)
        iscrowd = torch.tensor([obj.get("iscrowd", 0) for obj in anno], dtype=torch.int64)
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
