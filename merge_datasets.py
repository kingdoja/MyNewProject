import os
import json
import shutil
from pathlib import Path

def merge_datasets(dataset_names, source_base_path, target_base_path):
    """
    合并多个数据集及其标注文件
    
    Args:
        dataset_names: 要合并的数据集名称列表
        source_base_path: 源数据集的基础路径
        target_base_path: 目标合并数据集的路径
    """
    
    # 创建目标目录
    target_images_path = Path(target_base_path) / "images"
    target_images_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化合并后的JSON结构
    merged_data = {
        "info": {
            "description": "Merged dataset from multiple patches",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # 用于跟踪ID映射，避免冲突
    image_id_offset = 0
    annotation_id_offset = 0
    category_mapping = {}
    category_id_counter = 1
    
    # 遍历所有数据集
    for dataset_name in dataset_names:
        dataset_path = Path(source_base_path) / dataset_name
        images_path = dataset_path / "images"
        json_path = dataset_path / f"{dataset_name}.json"
        
        # 检查数据集是否存在
        if not dataset_path.exists():
            print(f"Warning: Dataset {dataset_name} not found at {dataset_path}")
            continue
            
        if not json_path.exists():
            print(f"Warning: JSON file not found at {json_path}")
            continue
        
        # 读取原始JSON文件
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 处理categories（如果还没有添加）
        if not merged_data["categories"]:
            merged_data["categories"] = data["categories"]
            # 建立类别映射
            for cat in data["categories"]:
                category_mapping[cat["id"]] = cat["id"]
        else:
            # 如果已有类别，需要检查是否需要合并
            existing_categories = {cat["name"]: cat["id"] for cat in merged_data["categories"]}
            for cat in data["categories"]:
                if cat["name"] not in existing_categories:
                    # 添加新类别
                    new_cat_id = category_id_counter
                    category_id_counter += 1
                    category_mapping[cat["id"]] = new_cat_id
                    new_category = cat.copy()
                    new_category["id"] = new_cat_id
                    merged_data["categories"].append(new_category)
                else:
                    category_mapping[cat["id"]] = existing_categories[cat["name"]]
        
        # 复制图片并更新JSON中的images信息
        image_id_mapping = {}
        for img_info in data["images"]:
            original_image_id = img_info["id"]
            new_image_id = original_image_id + image_id_offset
            
            # 更新图片信息
            new_img_info = img_info.copy()
            new_img_info["id"] = new_image_id
            # 更新文件名以避免冲突
            new_filename = f"{dataset_name}_{img_info['file_name']}"
            new_img_info["file_name"] = new_filename
            
            merged_data["images"].append(new_img_info)
            image_id_mapping[original_image_id] = new_image_id
            
            # 复制图片文件
            src_image_path = images_path / img_info["file_name"]
            dst_image_path = target_images_path / new_filename
            
            if src_image_path.exists():
                shutil.copy2(src_image_path, dst_image_path)
            else:
                print(f"Warning: Image file not found: {src_image_path}")
        
        # 更新annotations信息
        for ann in data["annotations"]:
            new_ann = ann.copy()
            # 更新annotation ID
            new_ann["id"] = ann["id"] + annotation_id_offset
            # 更新image ID
            new_ann["image_id"] = image_id_mapping[ann["image_id"]]
            # 更新category ID
            new_ann["category_id"] = category_mapping[ann["category_id"]]
            
            merged_data["annotations"].append(new_ann)
        
        # 更新偏移量
        if data["images"]:
            image_id_offset += max([img["id"] for img in data["images"]]) + 1
        if data["annotations"]:
            annotation_id_offset += max([ann["id"] for ann in data["annotations"]]) + 1
    
    # 保存合并后的JSON文件
    merged_json_path = Path(target_base_path) / "merged_dataset.json"
    with open(merged_json_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged dataset saved to {target_base_path}")
    print(f"Total images: {len(merged_data['images'])}")
    print(f"Total annotations: {len(merged_data['annotations'])}")
    print(f"Total categories: {len(merged_data['categories'])}")

if __name__ == "__main__":
    # 定义要合并的数据集
    datasets_to_merge = ["patches3", "patches4", "patches212", "patches182", "patchesK265"]
    
    # 设置源路径和目标路径
    source_path = "/path/to/source/datasets"  # 修改为实际源数据集路径
    target_path = "/home/ubuntu/lsn/project_new/RT-DETR-main/merged_dataset"  # 合并后数据集的目标路径
    
    # 执行合并操作
    merge_datasets(datasets_to_merge, source_path, target_path)