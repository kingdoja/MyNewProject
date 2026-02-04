import json
import os

def generate_classes_txt_from_coco(json_file_path, output_file_path):
    """
    从COCO格式的JSON文件中提取类别名称并生成classes.txt文件
    
    Args:
        json_file_path: COCO格式的JSON文件路径
        output_file_path: 输出的classes.txt文件路径
    """
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 提取类别名称
    categories = data.get('categories', [])
    
    # 按照ID排序以确保顺序正确
    categories.sort(key=lambda x: x['id'])
    
    # 提取类别名称
    class_names = [category['name'] for category in categories]
    
    # 写入classes.txt文件
    with open(output_file_path, 'w') as f:
        for class_name in class_names:
            f.write(class_name + '\n')
    
    print(f"成功生成 {output_file_path}")
    print(f"类别数量: {len(class_names)}")
    print("类别列表:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

def generate_classes_txt_from_splits(base_dir):
    """
    从划分后的数据集中生成classes.txt文件
    
    Args:
        base_dir: 划分后的数据集根目录
    """
    # 查找任何一个分割集的标注文件（通常所有分割集的类别都相同）
    for split in ['train', 'val', 'test']:
        annotations_dir = os.path.join(base_dir, split, 'annotations')
        if os.path.exists(annotations_dir):
            for file in os.listdir(annotations_dir):
                if file.startswith('instances_') and file.endswith('.json'):
                    json_file_path = os.path.join(annotations_dir, file)
                    output_file_path = os.path.join(base_dir, 'classes.txt')
                    generate_classes_txt_from_coco(json_file_path, output_file_path)
                    return
    
    print("未找到任何标注文件")

# 使用示例
if __name__ == "__main__":
    # 方式1: 直接从原始JSON文件生成
    # json_file_path = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/patches212/result.json"
    # output_file_path = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/classes.txt"
    # generate_classes_txt_from_coco(json_file_path, output_file_path)
    
    # 方式2: 从划分后的数据集生成
    base_dir = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset"
    generate_classes_txt_from_splits(base_dir)