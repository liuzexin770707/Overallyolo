from ultralytics import YOLO
import os

# ================= 1. 训练参数自定义配置区 =================
MODEL_WEIGHTS = 'yolov8n.pt'            # 预训练模型，首次运行会自动下载。可选 yolov8s.pt, yolov8m.pt 等
DATA_YAML = './yolo_dataset/dataset.yaml' # 刚才脚本1生成的 yaml 文件路径
EPOCHS = 100                            # 训练轮数 (建议初测用50-100，正式用300)
BATCH_SIZE = 16                         # 批次大小 (如果显存爆了报错 CUDA out of memory，请将其调小为 8 或 4)
IMG_SIZE = 320                          # 输入图片大小
PROJECT_NAME = 'yolo_runs'              # 训练结果保存的主文件夹名
RUN_NAME = 'train_exp_1'                # 本次训练结果的子文件夹名
WORKERS = 4                             # 数据加载的多线程数 (Windows如果报错建议改为 0)
DEVICE = '0'                             # 运行设备: '' 表示自动选择(优先GPU)，也可指定 '0' (第一张GPU) 或 'cpu'
# ===========================================================

def main():
    # 检查 yaml 文件是否存在
    if not os.path.exists(DATA_YAML):
        print(f"❌ 错误: 找不到配置文件 {DATA_YAML}，请确认脚本1是否成功运行！")
        return

    print("🚀 开始初始化 YOLOv8 模型...")
    # 加载预训练模型
    model = YOLO(MODEL_WEIGHTS)

    print(f"🔥 开始训练... 将进行 {EPOCHS} 轮训练")
    # 开始训练
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        workers=WORKERS,
        device=DEVICE,
        # 下面是一些进阶参数，你可以根据需要取消注释并修改：
        # patience=50,       # 早停机制 (如果连续50轮没有提升则停止)
        # save=True,         # 是否保存训练好的模型
        # val=True,          # 是否在训练期间进行验证
    )

    print("\n" + "="*40)
    print("✅ 训练结束！")
    print(f"📁 模型权重 (best.pt 和 last.pt) 和可视化结果已保存在: ./{PROJECT_NAME}/{RUN_NAME}/weights/")
    print("="*40)

if __name__ == '__main__':
    # Windows 环境下多线程训练必须放在 if __name__ == '__main__': 下面
    main()