import os
import random
import shutil

# ================= 1. 自定义配置区 =================
SRC_DIR = './dataAll'              # 源文件夹路径 (需要包含 images 和 labels 子文件夹)
DST_DIR = './yolo_dataset'         # 目标文件夹路径
CHUNK_SIZE = 10                    # 分组大小 (例如连续 10 个样本为一组)
VAL_NUM_PER_CHUNK = 2              # 每组中抽取多少个作为校验集 (val)
# ===================================================

def create_dirs(base_path):
    """创建YOLO所需的文件目录结构"""
    dirs = [
        os.path.join(base_path, 'images/train'),
        os.path.join(base_path, 'images/val'),
        os.path.join(base_path, 'labels/train'),
        os.path.join(base_path, 'labels/val')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def generate_yaml_template(dst_dir, classes_file):
    """生成 dataset.yaml 模板"""
    yaml_path = os.path.join(dst_dir, 'dataset.yaml')
    
    # 尝试读取 classes.txt 获取类别名
    classes = []
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    
    # 如果没找到 classes.txt，给一个默认提示
    if not classes:
        classes = ['class0', 'class1', 'class2', '... # 请手动替换为你的真实类别名']

    yaml_content = f"""# YOLOv8 Dataset Configuration
path: {os.path.abspath(dst_dir)} # 数据集根目录的绝对路径 (建议保持绝对路径)
train: images/train # 训练集图片相对路径
val: images/val     # 验证集图片相对路径

# 类别信息
names:
"""
    for i, cls_name in enumerate(classes):
        yaml_content += f"  {i}: {cls_name}\n"

    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    return yaml_path

def main():
    print("开始处理数据集...")
    src_images_dir = os.path.join(SRC_DIR, 'images')
    src_labels_dir = os.path.join(SRC_DIR, 'labels')
    classes_file = os.path.join(src_labels_dir, 'classes.txt')

    create_dirs(DST_DIR)

    # 获取所有图片并排序，保证连续性
    all_images = sorted([f for f in os.listdir(src_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    train_count = 0
    val_count = 0
    missing_labels = 0

    # 按块 (Chunk) 遍历图片
    for i in range(0, len(all_images), CHUNK_SIZE):
        chunk = all_images[i:i + CHUNK_SIZE]
        
        # 过滤出有对应标签的图片，防止孤立图片报错
        valid_chunk = []
        for img in chunk:
            label_name = os.path.splitext(img)[0] + '.txt'
            if os.path.exists(os.path.join(src_labels_dir, label_name)):
                valid_chunk.append(img)
            else:
                print(f"⚠️ 警告: 找不到 {img} 的对应标签文件，已跳过。")
                missing_labels += 1

        # 在当前有效块中随机选择验证集
        val_samples = random.sample(valid_chunk, min(VAL_NUM_PER_CHUNK, len(valid_chunk)))
        
        for img in valid_chunk:
            is_val = img in val_samples
            subset = 'val' if is_val else 'train'
            
            if is_val: val_count += 1
            else: train_count += 1

            # 拷贝图片和标签
            label_name = os.path.splitext(img)[0] + '.txt'
            
            shutil.copy2(os.path.join(src_images_dir, img), 
                         os.path.join(DST_DIR, f'images/{subset}', img))
            shutil.copy2(os.path.join(src_labels_dir, label_name), 
                         os.path.join(DST_DIR, f'labels/{subset}', label_name))

    # 生成 YAML
    yaml_path = generate_yaml_template(DST_DIR, classes_file)

    # 输出总结信息
    print("\n" + "="*40)
    print("🎉 数据集划分完成！")
    print(f"📁 输出目录: {os.path.abspath(DST_DIR)}")
    print(f"📊 训练集 (Train) 数量: {train_count} 份")
    print(f"📊 校验集 (Val) 数量: {val_count} 份")
    if missing_labels > 0:
        print(f"⚠️ 剔除无标签样本数: {missing_labels} 份")
    print("\n⚠️ 【下一步重要提醒】 ⚠️")
    print(f"1. 请打开生成的配置文件：{yaml_path}")
    print("2. 检查 `names` 下的类别名称是否与你标注时的一致。如果之前存在 `classes.txt`，脚本已经自动帮你填入了，但仍建议核对。")
    print("="*40)

if __name__ == '__main__':
    main()