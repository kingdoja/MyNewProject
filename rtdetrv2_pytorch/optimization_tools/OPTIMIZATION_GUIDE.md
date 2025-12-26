# RT-DETR 痰液细胞检测模型优化指南

## 问题诊断流程

### 1. 数据集诊断

首先运行数据集诊断脚本，检查数据质量：

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch
python optimization_tools/diagnose_dataset.py \
    --ann-file /home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug/train/annotations/instances_train.json \
    --img-folder /home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug/train/images \
    --output-dir ./diagnosis_output
```

**检查要点：**
- ✅ 类别分布是否平衡（不平衡度 < 10）
- ✅ 小目标占比（< 50%为佳）
- ✅ 标注质量（无效框、超出边界框）
- ✅ 每张图像的平均标注数

### 2. 训练监控

训练过程中实时监控：

```bash
# 在训练过程中，定期运行监控脚本
python optimization_tools/monitor_training.py \
    --log-file ./logs/training.log \
    --output-dir ./training_monitor
```

**关注指标：**
- Loss是否正常下降（前10个epoch应下降 > 30%）
- 验证AP是否提升
- 是否出现过拟合（训练loss下降但验证AP下降）

## 常见问题及解决方案

### 问题1: 精度和召回率都很低

**可能原因：**
1. **num_queries不足** - 对于5万张图像和12个类别，300个查询可能不够
2. **小目标占比高** - 细胞目标通常较小，需要更多查询
3. **数据增强过度** - 过度增强可能破坏小目标

**解决方案：**
```yaml
# 在配置文件中增加num_queries
RTDETRTransformerv2:
  num_queries: 600  # 从300增加到600

RTDETRPostProcessor:
  num_top_queries: 600  # 保持一致
```

### 问题2: 类别不平衡导致某些类别检测效果差

**解决方案：**
1. **启用类别权重**（如果已有权重文件）：
```yaml
RTDETRCriterionv2:
  class_weight_file: /path/to/class_weights.json
  class_weight_key: weight
  class_weight_power: 1.0
```

2. **使用加权采样器**：
```yaml
train_dataloader:
  sampler:
    type: CocoImageWeightedRandomSampler
    weights_file: /path/to/image_weights.json
    replacement: True
    default_weight: 1.0
```

3. **调整Focal Loss参数**：
```yaml
RTDETRCriterionv2:
  alpha: 0.3  # 提高alpha，更关注正样本
  gamma: 3.0  # 提高gamma，更关注难样本
```

### 问题3: 小目标检测效果差

**解决方案：**
1. **增加num_queries**（最重要）
2. **调整损失权重**：
```yaml
RTDETRCriterionv2:
  weight_dict: 
    loss_vfl: 6  # 提高分类损失权重
    loss_bbox: 2.5  # 提高定位损失权重
    loss_giou: 1.5  # 提高IoU损失权重
```

3. **减少数据增强强度**：
```yaml
# 降低可能破坏小目标的增强概率
- {type: RandomZoomOut, fill: 0, p: 0.05}  # 从0.1降到0.05
- {type: RandomIoUCrop, p: 0.2, min_scale: 0.85}  # 更温和的裁剪
```

4. **提高最小框尺寸阈值**：
```yaml
- {type: SanitizeBoundingBoxes, min_size: 2}  # 从1提高到2
```

### 问题4: 训练loss不下降或下降很慢

**可能原因：**
1. 学习率过小
2. 数据增强过度
3. 模型容量不足

**解决方案：**
```yaml
optimizer:
  lr: 0.0001  # 从0.00005提高到0.0001

lr_warmup_scheduler:
  warmup_duration: 3000  # 增加warmup步数
```

### 问题5: 过拟合

**解决方案：**
```yaml
Transformer:
  dropout: 0.15  # 适中的dropout率

# 使用早停机制
early_stopping_patience: 25
early_stopping_min_delta: 0.00005
```

## 优化配置对比

### 原始配置 vs 优化配置

| 参数 | 原始值 | 优化值 | 原因 |
|------|--------|--------|------|
| num_queries | 300 | 600 | 提高小目标召回率 |
| 基础学习率 | 0.00005 | 0.0001 | 加快收敛 |
| loss_vfl权重 | 5 | 6 | 更关注分类 |
| loss_bbox权重 | 2 | 2.5 | 更精确的定位 |
| loss_giou权重 | 1 | 1.5 | 更好的重叠度 |
| Focal Loss alpha | 0.25 | 0.3 | 更关注正样本 |
| Focal Loss gamma | 2.5 | 3.0 | 更关注难样本 |
| RandomZoomOut概率 | 0.1 | 0.05 | 减少小目标丢失 |
| RandomIoUCrop概率 | 0.3 | 0.2 | 减少过度裁剪 |
| min_size阈值 | 1 | 2 | 过滤极小框 |

## 训练建议

### 1. 分阶段训练策略

**阶段1: 前30个epoch**
- 使用较强的数据增强
- 较高的学习率
- 关注loss下降趋势

**阶段2: 30-70个epoch**
- 逐步减少数据增强强度
- 学习率衰减
- 关注验证指标提升

**阶段3: 70+ epoch**
- 最小数据增强
- 低学习率微调
- 使用早停机制

### 2. 超参数调优顺序

1. **首先调整num_queries**（影响最大）
2. **然后调整损失权重**
3. **最后调整学习率**

### 3. 验证集检查

确保验证集：
- 与训练集分布一致
- 包含所有类别
- 标注质量高

## 使用优化配置

使用优化后的配置文件进行训练：

```bash
python train_sputum_cell.py \
    -c configs/rtdetrv2/rtdetrv2_r50vd_cancer_detection1_optimized.yml \
    -d cuda:0
```

## 预期改进

使用优化配置后，预期改进：
- **AP提升**: 5-15%
- **AP50提升**: 8-20%
- **小目标召回率**: 提升10-25%
- **训练稳定性**: 显著提升

## 进一步优化方向

1. **使用更大的backbone**（如果显存允许）
2. **集成学习**（多个模型集成）
3. **测试时增强（TTA）**
4. **更精细的类别权重调整**
5. **使用预训练权重微调**

## 故障排查清单

- [ ] 数据集诊断通过
- [ ] 类别分布检查
- [ ] 标注质量检查
- [ ] 小目标占比分析
- [ ] 训练loss正常下降
- [ ] 验证指标提升
- [ ] 无过拟合现象
- [ ] 配置参数合理

## 联系与支持

如遇到问题，请检查：
1. 训练日志中的错误信息
2. 数据集诊断报告
3. 训练监控曲线

