from ultralytics import YOLO
import os

# ================= 训练参数自定义配置区 =================
# 【1. 模型权重与任务配置】
# 场景 A (⚠️全新训练)       : 填入官方预训练权重，例如 'yolo26n.pt'
# 场景 B (扩充数据集微调) : 填入你之前的 best.pt 路径，例如 './yolo_runs/train_exp_1/weights/best.pt'
# 场景 C (断点续训)       : 填入意外中断时的 last.pt 路径，例如 './yolo_runs/train_exp_1/weights/last.pt'
MODEL_WEIGHTS = r'.\runs\detect\yolo_runs\train_exp_2\weights\last.pt'            

# 是否开启断点续训？ 
# ⚠️ 注意：仅在场景 C (中途报错断电，想接着跑完剩余轮数) 时设为 True，且 MODEL_WEIGHTS 必须是 last.pt
# 如果是扩充数据集微调，这里必须保持 False！
RESUME_TRAINING = True                 

# 【2. 常规参数配置】
DATA_YAML = r'./yolo_dataset/dataset.yaml' # ⚠️数据集配置文件路径
EPOCHS = 600                            # 训练轮数 (如果是断点续训，系统会自动计算剩余轮数)
BATCH_SIZE = 64                         # 批次大小 (如果显存爆了请调小)
IMG_SIZE = 320                          # 输入图片大小
PROJECT_NAME = 'yolo_runs'              # 训练结果保存的主文件夹名
RUN_NAME = 'train_exp_3'                # 本次训练结果的子文件夹名 (⚠️建议每次新开训练改个名，避免覆盖)
WORKERS = 8                             # 数据加载的多线程数 (Windows报错建议改为 0)
DEVICE = '0'                            # 运行设备: '' 表示自动，也可指定 '0' (第一张GPU)
# ===========================================================

def main():
    # 检查 yaml 文件是否存在
    if not os.path.exists(DATA_YAML):
        print(f"❌ 错误: 找不到配置文件 {DATA_YAML}，请检查路径！")
        return

    print(f"🚀 开始初始化 YOLO 模型 (加载权重: {MODEL_WEIGHTS})...")
    # 加载模型
    model = YOLO(MODEL_WEIGHTS)

    if RESUME_TRAINING:
        print("🔄 检测到断点续训模式已开启，将自动读取 last.pt 的历史进度继续训练...")
    else:
        print(f"🔥 开始常规训练/微调... 目标总轮数: {EPOCHS}")

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
        resume=RESUME_TRAINING,  # <--- 核心新增参数：通过上方配置控制是否续训
        
        # 下面是一些进阶参数，可以根据需要取消注释并修改：
        # patience=50,       # 早停机制 (如果连续50轮没有提升则提前停止)
        # save=True,         # 是否保存训练好的模型
        # val=True,          # 是否在训练期间进行验证
    )

    print("\n" + "="*40)
    print("✅ 训练结束！")
    print(f"📁 模型权重 (best.pt 和 last.pt) 已保存在: ./{PROJECT_NAME}/{RUN_NAME}/weights/")
    print("="*40)

if __name__ == '__main__':
    # Windows 环境下多线程训练必须放在 if __name__ == '__main__': 下面
    main()