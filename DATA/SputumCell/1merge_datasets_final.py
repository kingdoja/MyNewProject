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
            "year": 2025,
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # 用于跟踪ID映射，避免冲突
    image_id_offset = 0
    annotation_id_offset = 0
    # 统一类别到连续ID（从0开始）的映射
    name_to_new_id = {}
    next_cat_id = 0
    # 每个源dataset的 老ID->新ID 映射会临时构造
    
    # 遍历所有数据集
    dataset_json_candidates = {
        "patches3": ["coco_format.json"],
        "patches4": ["coco_format.json"],
        "patches182": ["coco_format.json"],
        "patches212": ["coco_format.json"],
        "patchesK265": ["coco_format.json"],
        "new45": ["coco_format.json"],
        "new44": ["coco_format.json"],
        # "merged_dataset1": ["result.json"]
    }
    
    for dataset_idx, dataset_name in enumerate(dataset_names):
        dataset_path = Path(source_base_path) / dataset_name
        
        if not dataset_path.exists():
            print(f"Warning: Dataset {dataset_name} not found at {dataset_path}")
            continue
        
        possible_json_files = dataset_json_candidates.get(
            dataset_name,
            ["coco_format.json", f"{dataset_name}.json"]
        )
        
        json_path = None
        for candidate in possible_json_files:
            candidate_path = dataset_path / candidate
            if candidate_path.exists():
                json_path = candidate_path
                break
        if json_path is None:
            tried_files = ", ".join(possible_json_files)
            print(f"Warning: JSON file not found for {dataset_name}. Tried: {tried_files}")
            continue
        
        # 读取原始JSON文件
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        # 处理JSON数据，确保是正确的格式
        # 如果是列表格式，需要检查是否是多个COCO数据集的列表
        if isinstance(raw_data, list):
            # 检查列表中的元素是否是COCO格式（包含images、annotations等键）
            if raw_data and isinstance(raw_data[0], dict) and "images" in raw_data[0]:
                # 如果是COCO格式的列表，合并所有元素
                data = {
                    "images": [],
                    "annotations": [],
                    "categories": []
                }
                seen_cat_ids = {}
                cat_id_offset = 0
                ann_id_offset = 0
                img_id_offset = 0
                
                for item in raw_data:
                    if isinstance(item, dict) and "images" in item:
                        # 合并images
                        for img in item.get("images", []):
                            new_img = img.copy()
                            if "id" in new_img:
                                new_img["id"] = new_img["id"] + img_id_offset
                            data["images"].append(new_img)
                        
                        # 合并annotations（需要更新image_id和category_id）
                        for ann in item.get("annotations", []):
                            new_ann = ann.copy()
                            if "id" in new_ann:
                                new_ann["id"] = new_ann["id"] + ann_id_offset
                            if "image_id" in new_ann:
                                new_ann["image_id"] = new_ann["image_id"] + img_id_offset
                            data["annotations"].append(new_ann)
                        
                        # 合并categories（去重）
                        for cat in item.get("categories", []):
                            cat_id = cat.get("id")
                            if cat_id not in seen_cat_ids:
                                seen_cat_ids[cat_id] = len(data["categories"])
                                data["categories"].append(cat)
                        
                        # 更新偏移量
                        if item.get("images"):
                            max_img_id = max([img.get("id", 0) for img in item["images"]], default=0)
                            img_id_offset += max_img_id + 1
                        if item.get("annotations"):
                            max_ann_id = max([ann.get("id", 0) for ann in item["annotations"]], default=0)
                            ann_id_offset += max_ann_id + 1
            else:
                # 如果不是COCO格式的列表，尝试取第一个元素（向后兼容）
                print(f"Warning: JSON is a list but not in COCO format, trying first element")
                data = raw_data[0] if raw_data else {}
        else:
            data = raw_data
        
        # 确保必要的键存在
        if "images" not in data:
            data["images"] = []
        if "annotations" not in data:
            data["annotations"] = []
        if "categories" not in data:
            data["categories"] = []
        
        # 统一构建 本数据集 老ID->新ID 的映射（按name对齐）
        local_cat_id_map = {}
        for cat in data["categories"]:
            if not isinstance(cat, dict) or "id" not in cat or "name" not in cat:
                continue
            name = cat["name"]
            if name not in name_to_new_id:
                name_to_new_id[name] = next_cat_id
                next_cat_id += 1
            local_cat_id_map[cat["id"]] = name_to_new_id[name]
        
        # 复制图片并更新JSON中的images信息
        image_id_mapping = {}  # 原始ID -> 新ID的映射
        file_name_to_image_id = {}  # 文件名 -> 新图片ID的映射（用于处理重复文件）
        skipped_images = 0  # 统计跳过的图片数量（文件不存在）
        duplicate_images = 0  # 统计重复的图片数量（文件存在但已处理过）
        
        for img_info in data["images"]:
            if not isinstance(img_info, dict) or "id" not in img_info:
                continue
                
            original_image_id = img_info["id"]
            
            # 先尝试查找并复制文件，只有文件存在时才添加到合并结果
            if "file_name" not in img_info:
                continue
                
            original_file_name = img_info["file_name"]
            base_file_name = original_file_name.split('/')[-1]  # 提取基础文件名
            new_file_name = f"{dataset_name}_{base_file_name}"
            
            # 尝试多种可能的路径格式
            possible_paths = [
                dataset_path / original_file_name,  # 原始路径
                dataset_path / base_file_name,  # 仅文件名
                dataset_path / "images" / base_file_name,  # images子目录 + 文件名
                dataset_path / "images" / original_file_name,  # images子目录 + 原始路径
            ]
            
            src_image_path = None
            for path in possible_paths:
                if path.exists():
                    src_image_path = path
                    break
            
            dst_image_path = target_images_path / new_file_name
            
            # 只有文件存在时才处理
            if src_image_path and src_image_path.exists():
                # 检查是否已经处理过这个文件（重复文件）
                if new_file_name in file_name_to_image_id:
                    # 这是重复文件，但文件存在，需要合并标注框
                    # 将原始图片ID映射到已存在的图片ID
                    existing_image_id = file_name_to_image_id[new_file_name]
                    image_id_mapping[original_image_id] = existing_image_id
                    duplicate_images += 1
                    # 不重复复制文件，也不重复添加图片记录
                else:
                    # 新文件，正常处理
                    # 复制文件（如果目标文件已存在，会覆盖）
                    if not dst_image_path.exists():
                        shutil.copy2(src_image_path, dst_image_path)
                    
                    # 文件存在后才添加到合并结果
                    new_image_id = original_image_id + image_id_offset
                    new_img_info = img_info.copy()
                    new_img_info["id"] = new_image_id
                    new_img_info["file_name"] = new_file_name
                    
                    merged_data["images"].append(new_img_info)
                    image_id_mapping[original_image_id] = new_image_id
                    file_name_to_image_id[new_file_name] = new_image_id  # 记录文件名到ID的映射
            else:
                # 文件不存在，跳过该图片
                skipped_images += 1
                if skipped_images <= 10:  # 只打印前10个警告，避免输出过多
                    print(f"Warning: Image file not found, skipping: {original_file_name}")
                elif skipped_images == 11:
                    print(f"Warning: ... (more missing files, total {skipped_images} so far)")
        
        if skipped_images > 0:
            print(f"Info: Skipped {skipped_images} images from {dataset_name} due to missing files")
        if duplicate_images > 0:
            print(f"Info: Merged annotations from {duplicate_images} duplicate images in {dataset_name}")
        
        # 更新annotations信息
        for ann in data["annotations"]:
            if not isinstance(ann, dict) or "id" not in ann or "image_id" not in ann or "category_id" not in ann:
                continue
                
            new_ann = ann.copy()
            # 更新annotation ID
            new_ann["id"] = ann["id"] + annotation_id_offset
            # 更新image ID
            if ann["image_id"] in image_id_mapping:
                new_ann["image_id"] = image_id_mapping[ann["image_id"]]
            # 更新category ID（按统一映射）
            if ann["category_id"] in local_cat_id_map:
                new_ann["category_id"] = local_cat_id_map[ann["category_id"]]
            else:
                # 找不到映射，说明该类别未在categories中定义，跳过该标注
                continue
            
            merged_data["annotations"].append(new_ann)
        
        # 更新偏移量
        if data["images"]:
            image_ids = [img["id"] for img in data["images"] if isinstance(img, dict) and "id" in img]
            if image_ids:
                image_id_offset += max(image_ids) + 10000
        if "annotations" in data and data["annotations"]:
            ann_ids = [ann["id"] for ann in data["annotations"] if isinstance(ann, dict) and "id" in ann]
            if ann_ids:
                annotation_id_offset += max(ann_ids) + 10000
    
    # 统一写入规范化的categories（按ID升序）
    merged_data["categories"] = [{"id": cid, "name": name} for name, cid in name_to_new_id.items()]
    merged_data["categories"].sort(key=lambda x: x["id"]) 

    # 保存合并后的JSON文件
    merged_json_path = Path(target_base_path) / "result.json"
    with open(merged_json_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged dataset saved to {target_base_path}")
    print(f"Total images: {len(merged_data['images'])}")
    print(f"Total annotations: {len(merged_data['annotations'])}")
    print(f"Total categories: {len(merged_data['categories'])}")

if __name__ == "__main__":
    # 定义要合并的数据集
    # datasets_to_merge = ["patches3", "patches4", "patches182", "patches212", "patchesK265", "new45", "new44", "new37"]
    datasets_to_merge = ["new45", "new44", "new37"]
    
    # 设置源路径和目标路径
    source_path = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell"  # 修改为实际源数据集路径
    target_path = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/merged_dataset"  # 合并后数据集的目标路径
    
    # 执行合并操作
    merge_datasets(datasets_to_merge, source_path, target_path)