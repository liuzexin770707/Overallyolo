from pathlib import Path
import cv2
import imghdr

img_dir = Path('./yolo_dataset/images')
bad = 0
total = 0

for f in sorted(img_dir.rglob('*')):
    if f.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}:
        continue
    total += 1
    # 方法1: imghdr 检查
    typ = imghdr.what(str(f))
    # 方法2: OpenCV 实际读取
    img = cv2.imread(str(f))
    if img is None:
        print(f'❌ 损坏/无法读取: {f}')
        bad += 1

print(f'\n检查完成: 共 {total} 张图片, {bad} 张损坏')
