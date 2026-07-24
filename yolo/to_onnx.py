from ultralytics import YOLO
import os
#####不要改变任何参数，只需要运行即可

def export_yolov8n_to_onnx(pt_path, output_dir=None):
    """
    将 YOLOv8n 的 .pt 模型导出为 ONNX 格式，优化用于 K230 部署。

    参数：
    - pt_path: 训练好的 .pt 模型路径
    - output_dir: ONNX 输出目录（可选，默认与 .pt 同目录）
    """
    try:
        # 加载模型
        model = YOLO(pt_path)

        # 获取默认输出路径（与 .pt 同目录）
        if output_dir is None:
            output_dir = os.path.dirname(pt_path)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 导出 ONNX 文件
        onnx_path = model.export(
            format="onnx",  # 导出格式为 ONNX
            imgsz=320,  # 输入尺寸，与训练一致
            simplify=True,  # 简化模型，减少冗余节点
            opset=12,  # 指定 ONNX Opset 版本，兼容 K230 的 nncase
            dynamic=False,  # 固定输入尺寸（K230 不支持动态形状）
            half=False  # 不使用 FP16（保持 FP32，K230 需要 INT8 量化）
        )

        print(f"ONNX 文件已成功导出至: {onnx_path}")

        # 验证 ONNX 文件
        verify_onnx(onnx_path)

    except Exception as e:
        print(f"导出失败: {str(e)}")
        raise


def verify_onnx(onnx_path):
    """
    验证 ONNX 文件的输入输出形状。
    """
    try:
        import onnxruntime as ort
        import numpy as np

        # 加载 ONNX 文件
        session = ort.InferenceSession(onnx_path)

        # 获取输入和输出信息
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # 打印输入输出形状
        print("输入信息:", session.get_inputs()[0])
        print("输出信息:", session.get_outputs()[0])

        # 测试推理（随机输入）
        dummy_input = np.random.randn(1, 3, 320, 320).astype(np.float32)
        outputs = session.run([output_name], {input_name: dummy_input})
        print(f"输出形状: {outputs[0].shape}")

        # 预期 YOLOv8n 输出形状（假设 nc=80，COCO 数据集）
        expected_shape = (1, 5, 2100)  # nc + 5（边界框 + 置信度）
        if outputs[0].shape != expected_shape:
            print(f"警告：输出形状 {outputs[0].shape} 与预期 {expected_shape} 不匹配，请检查类别数或导出参数！")

    except Exception as e:
        print(f"验证失败: {str(e)}")
        raise


if __name__ == '__main__':
    # 指定你的 .pt 模型路径
    pt_path = r"C:\Users\30324\Desktop\yolo\best.pt"

    # 导出 ONNX
    export_yolov8n_to_onnx(pt_path)