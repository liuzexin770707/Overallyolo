from libs.PipeLine import PipeLine, ScopedTiming
from libs.YOLO import YOLOv8
from libs.Utils import *
from media.sensor import *
import os, sys, gc
import ulab.numpy as np
import image

if __name__ == "__main__":
    # ============================================================================
    # 1. 核心参数配置
    # ============================================================================
    rgb888p_size = [640, 480]
    display_size = [640, 480]
    model_input_size = [320, 320]

    kmodel_path = "/sdcard/best.kmodel"
    labels = ["tar"] # 你的模型类别列表

    # ============================================================================
    # 2. 硬件与 Pipeline 初始化
    # ============================================================================
    sensor = Sensor(width=1280, height=960)
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode="lcd")
    pl.create(sensor=sensor)

    # ============================================================================
    # 3. YOLO 模型初始化
    # ============================================================================
    yolo = YOLOv8(task_type="detect", mode="video", kmodel_path=kmodel_path, labels=labels,
                  rgb888p_size=rgb888p_size, model_input_size=model_input_size,
                  display_size=display_size, conf_thresh=0.5, nms_thresh=0.45)
    yolo.config_preprocess()

    print("🚀 核心识图程序已启动...")

    # ============================================================================
    # 4. 主程序循环
    # ============================================================================
    try:
        while True:
            os.exitpoint() # 允许在 CanMV IDE 中响应中断停止

            # 1. 获取摄像头当前帧图像
            img = pl.get_frame()

            # 2. 将图像送入模型进行推理，返回结果
            res = yolo.run(img)

            # 3. 调用 YOLO 库内置的绘制功能，将检测框画在 OSD 涂层上
            yolo.draw_result(res, pl.osd_img)

            # 4. 刷新屏幕显示画面
            pl.show_image()

            # 5. 垃圾回收，防止内存溢出
            gc.collect()

    except Exception as e:
        sys.print_exception(e)
    finally:
        # 释放 KPU 资源与 Pipeline 资源
        yolo.deinit()
        pl.destroy()
        print("🛑 程序已安全退出并释放资源。")
