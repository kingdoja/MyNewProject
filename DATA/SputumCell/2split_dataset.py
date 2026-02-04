import json
import os
import random
import shutil
from collections import defaultdict

def split_dataset(json_file_path, image_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    将SputumCell数据集按照指定比例划分为训练集、验证集和测试集
    
    Args:
        json_file_path: 原始JSON文件路径
        image_dir: 图片文件夹路径
        output_dir: 输出文件夹路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'annotations'), exist_ok=True)
    
    # 读取原始JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 按文件名分组，避免同一图片出现在不同集合中
    images_by_name = defaultdict(list)
    for image_info in data['images']:
        images_by_name[image_info['file_name']].append(image_info)
    
    # 获取唯一的文件名列表
    unique_filenames = list(images_by_name.keys())
    
    # 随机打乱文件名列表
    random.seed(42)  # 固定随机种子以确保可重复性
    random.shuffle(unique_filenames)
    
    # 计算划分点
    total_files = len(unique_filenames)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    # 划分文件名
    train_files = unique_filenames[:train_count]
    val_files = unique_filenames[train_count:train_count + val_count]
    test_files = unique_filenames[train_count + val_count:]
    
    # 创建文件名到集合的映射
    file_to_split = {}
    for file in train_files:
        file_to_split[file] = 'train'
    for file in val_files:
        file_to_split[file] = 'val'
    for file in test_files:
        file_to_split[file] = 'test'
    
    # 为每个集合创建JSON数据
    splits_data = {'train': {'images': [], 'annotations': []}, 
                   'val': {'images': [], 'annotations': []}, 
                   'test': {'images': [], 'annotations': []}}
    
    # 构建image_id到file_name的映射，便于后续查找
    image_id_to_file_name = {}
    for image_info in data['images']:
        image_id_to_file_name[image_info['id']] = image_info['file_name']
    
    # 为images分配ID并复制文件
    image_id_mapping = {}  # 原始ID到新ID的映射
    new_image_id = 1  # COCO格式通常从1开始
    
    copied_images_count = {'train': 0, 'val': 0, 'test': 0}
    
    for file_name, split_name in file_to_split.items():
        for original_image_info in images_by_name[file_name]:
            # 更新image信息
            new_image_info = original_image_info.copy()
            original_id = original_image_info['id']
            new_image_info['id'] = new_image_id
            image_id_mapping[original_id] = new_image_id
            
            # 添加到对应集合
            splits_data[split_name]['images'].append(new_image_info)
            
            # 复制图片文件
            src_path = os.path.join(image_dir, file_name)
            dst_path = os.path.join(output_dir, split_name, 'images', file_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_images_count[split_name] += 1
            else:
                print(f"警告: 图片文件不存在 {src_path}")
            
            new_image_id += 1
    
    # 处理annotations（如果有的话）
    annotation_count = {'train': 0, 'val': 0, 'test': 0}
    # 构建合法类别ID集合，过滤脏类别
    valid_cat_ids = set()
    if 'categories' in data:
        for c in data['categories']:
            if isinstance(c, dict) and 'id' in c:
                valid_cat_ids.add(c['id'])

    if 'annotations' in data and data['annotations']:
        for ann in data['annotations']:
            # 过滤非法类别
            if 'category_id' in ann and valid_cat_ids and ann['category_id'] not in valid_cat_ids:
                continue
            # 获取该annotation对应的图像文件名
            image_id = ann['image_id']
            if image_id in image_id_to_file_name:
                file_name = image_id_to_file_name[image_id]
                if file_name in file_to_split:
                    split_name = file_to_split[file_name]
                    # 更新annotation信息
                    new_ann = ann.copy()
                    new_ann['image_id'] = image_id_mapping[image_id]
                    new_ann['id'] = len(splits_data[split_name]['annotations']) + 1  # 为annotation分配新的ID
                    splits_data[split_name]['annotations'].append(new_ann)
                    annotation_count[split_name] += 1
    
    # 为每个集合生成JSON文件
    for split_name, split_data in splits_data.items():
        # 创建完整的COCO格式数据
        output_data = {
            'images': split_data['images'],
            'annotations': split_data['annotations'] if 'annotations' in data else [],
        }
        # 复制其他必要字段
        for key in ['categories', 'info', 'licenses']:
            if key in data:
                output_data[key] = data[key]
        
        # 写入JSON文件
        output_json_path = os.path.join(output_dir, split_name, 'annotations', f'instances_{split_name}.json')
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    # 输出统计信息
    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_files)} 个唯一文件, {len(splits_data['train']['images'])} 张图片 (成功复制: {copied_images_count['train']} 张)")
    print(f"  验证集: {len(val_files)} 个唯一文件, {len(splits_data['val']['images'])} 张图片 (成功复制: {copied_images_count['val']} 张)")
    print(f"  测试集: {len(test_files)} 个唯一文件, {len(splits_data['test']['images'])} 张图片 (成功复制: {copied_images_count['test']} 张)")
    print(f"  总计: {len(unique_filenames)} 个唯一文件")
    
    if 'annotations' in data and data['annotations']:
        print(f"  注释信息: 训练集{annotation_count['train']}个, 验证集{annotation_count['val']}个, 测试集{annotation_count['test']}个")

if __name__ == "__main__":
    # 设置路径
    # 使用 patches4 数据集进行划分
    json_file_path = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/merged_dataset/result.json"
    image_dir = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/merged_dataset/images"
    # 建议使用单独的输出目录，避免和之前的划分结果混在一起
    output_dir = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset"
    # json_file_path = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/new45/coco_format.json"
    # image_dir = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/new45/images"
    # # 建议使用单独的输出目录，避免和之前的划分结果混在一起
    # output_dir = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split45"
    
    # 执行划分
    split_dataset(json_file_path, image_dir, output_dir, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05)