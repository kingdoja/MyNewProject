"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from typing import Dict 

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from ...core import register


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.alpha
        gamma = self.gamma
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        
        # 添加边界检查，确保tgt_ids在有效范围内
        # 检查是否有任何目标标签ID超出类别数范围
        if len(tgt_ids) > 0:
            max_label = tgt_ids.max().item() if isinstance(tgt_ids, torch.Tensor) else max(tgt_ids)
            num_classes = out_prob.shape[1]
            if max_label >= num_classes:
                raise ValueError(f"目标标签ID {max_label} 超出类别数范围 {num_classes}。请检查数据集标注文件中的类别ID是否正确，确保所有标签ID都在 [0, {num_classes-1}] 范围内。")
            
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return {'indices': indices}
        
class DynamicLabelAssignment(nn.Module):
    """
    动态标签分配策略 (DLA)
    根据目标特征动态分配正负样本
    """
    def __init__(self, 
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 dynamic_threshold: float = 0.5):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.dynamic_threshold = dynamic_threshold

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 计算匹配成本
        indices = []
        for i in range(bs):
            out_prob = outputs["pred_logits"][i].softmax(-1)  # [num_queries, num_classes]
            out_bbox = outputs["pred_boxes"][i]  # [num_queries, 4]

            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]

            # 分类成本
            cost_class = -out_prob[:, tgt_ids]

            # 边框成本
            cost_bbox = torch.cdist(out_bbox, bbox_xyxy_to_cxcywh(tgt_bbox), p=1)

            # GIoU成本
            cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

            # 总成本矩阵
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(num_queries, -1).cpu()

            # 使用匈牙利算法进行匹配
            indices.append(linear_sum_assignment(C))

        # 动态调整阈值
        # 根据预测置信度和IoU动态调整正样本阈值
        dynamic_indices = []
        for i, (src, tgt) in enumerate(indices):
            # 获取匹配的预测框和真实框
            pred_boxes = outputs["pred_boxes"][i][src]
            true_boxes = targets[i]["boxes"]
            
            # 计算IoU
            iou_values = box_iou(pred_boxes, true_boxes)
            
            # 根据IoU动态调整阈值
            if len(iou_values) > 0:
                mean_iou = iou_values.diag().mean()
                # 根据平均IoU调整阈值
                adjusted_threshold = self.dynamic_threshold * (0.5 + 0.5 * mean_iou)
            else:
                adjusted_threshold = self.dynamic_threshold
            
            # 过滤低质量匹配
            valid_matches = iou_values.diag() >= adjusted_threshold
            src_filtered = src[valid_matches]
            tgt_filtered = tgt[valid_matches]
            
            dynamic_indices.append((src_filtered, tgt_filtered))
            
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in dynamic_indices]
