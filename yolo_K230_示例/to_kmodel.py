default_model=r".\runs\detect\yolo_runs\train_exp_3-2\weights\best.onnx" # onnx模型文件
default_dataset=r".\yolo_dataset\images\val" # 校准数据集路径
ptq_samples_count=95 # 校准数据集数量
default_ptq_option=0 # 量化等级选项(可选0,1,2,3,4,5)

#################增加了详细的调试信息输出
import os
import argparse
import numpy as np
from PIL import Image
import onnxsim
import onnx
import nncase
import shutil
import math
import onnx.helper
import time # 用于计算耗时

def parse_model_input_output(model_file,input_shape):
    onnx_model = onnx.load(model_file)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))
    input_tensors = [
        node for node in onnx_model.graph.input if node.name in input_names]

    # input
    inputs = []
    for _, e in enumerate(input_tensors):
        onnx_type = e.type.tensor_type
        input_dict = {}
        input_dict['name'] = e.name
        input_dict['dtype'] = onnx.helper.tensor_dtype_to_np_dtype(onnx_type.elem_type)
        input_dict['shape'] = [(i.dim_value if i.dim_value != 0 else d) for i, d in zip(
            onnx_type.shape.dim, input_shape)]
        inputs.append(input_dict)

    return onnx_model, inputs


def onnx_simplify(model_file, dump_dir,input_shape):
    print("  [1/2] 解析 ONNX 模型输入/输出节点...")
    onnx_model, inputs = parse_model_input_output(model_file,input_shape)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    input_shapes = {}
    for input in inputs:
        input_shapes[input['name']] = input['shape']

    print("  [2/2] 开始简化 ONNX 计算图 (onnxsim)... 这可能需要几秒钟。")
    onnx_model, check = onnxsim.simplify(onnx_model, overwrite_input_shapes=input_shapes)
    assert check, "Simplified ONNX model could not be validated"

    model_file = os.path.join(dump_dir, 'simplified.onnx')
    onnx.save_model(onnx_model, model_file)
    print("  -> 简化成功！")
    return model_file


def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content


def generate_data(shape, batch, calib_dir):
    print(f"  -> 开始读取并预处理量化校准图片... (目标数量: {batch})")
    img_paths = [os.path.join(calib_dir, p) for p in os.listdir(calib_dir)]
    data = []
    for i in range(batch):
        assert i < len(img_paths), f"严重错误: 提供的校准图片不足！需要 {batch} 张，仅找到 {len(img_paths)} 张。"
        if i % 10 == 0 or i == batch - 1:
            print(f"    处理图片进度: {i+1}/{batch}")
        
        img_data = Image.open(img_paths[i]).convert('RGB')
        img_data = img_data.resize((shape[3], shape[2]), Image.BILINEAR)
        img_data = np.asarray(img_data, dtype=np.uint8)
        img_data = np.transpose(img_data, (2, 0, 1))
        data.append([img_data[np.newaxis, ...]])
    print("  -> 图片加载完成！")
    return np.array(data)

def main():
    start_time = time.time()
    print("="*60)
    print("🔥 开始 nncase KModel 编译流程 🔥")
    print("="*60)

    parser = argparse.ArgumentParser(prog="nncase")
    parser.add_argument("--target", default="k230",type=str, help='target to run,k230/cpu')
    parser.add_argument("--model", default=default_model,type=str, help='model file')
    parser.add_argument("--dataset", default=default_dataset, type=str, help='calibration_dataset')
    parser.add_argument("--input_width", type=int, default=320, help='input_width')
    parser.add_argument("--input_height", type=int, default=320, help='input_height')
    parser.add_argument("--ptq_option", type=int, default=default_ptq_option, help='ptq_option:0,1,2,3,4')

    args = parser.parse_args()

    # 更新参数为32倍数
    input_width = int(math.ceil(args.input_width / 32.0)) * 32
    input_height = int(math.ceil(args.input_height / 32.0)) * 32

    # 模型的输入shape，维度要跟input_layout一致
    input_shape=[1,3,input_height,input_width]
    
    print(f"\n[STEP 1] 核心参数信息:")
    print(f"  - 目标硬件: {args.target}")
    print(f"  - 模型路径: {args.model}")
    print(f"  - 输入尺寸: {input_width}x{input_height} (已按32对齐)")
    print(f"  - 量化选项 (PTQ Option): {args.ptq_option}")
    print(f"  - 校准集数量: {ptq_samples_count}")

    dump_dir = 'tmp'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    print("\n[STEP 2] 准备与简化 ONNX 模型...")
    model_file = onnx_simplify(args.model, dump_dir,input_shape)

    print("\n[STEP 3] 配置 nncase 编译器...")
    compile_options = nncase.CompileOptions()
    compile_options.target = args.target
    compile_options.preprocess = True
    compile_options.swapRB = False
    compile_options.input_shape = input_shape
    compile_options.input_type = 'uint8'
    compile_options.input_range = [0, 1]
    compile_options.mean = [0, 0, 0]
    compile_options.std = [1, 1, 1]
    compile_options.input_layout = "NCHW"
    compile_options.quant_type = 'uint8'

    compiler = nncase.Compiler(compile_options)
    
    print("  -> 导入 ONNX 模型到 nncase...")
    model_content = read_model_file(model_file)
    import_options = nncase.ImportOptions()
    compiler.import_onnx(model_content, import_options)
    print("  -> 导入成功！")

    print("\n[STEP 4] 配置 PTQ 量化参数并加载数据集...")
    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = ptq_samples_count

    if args.ptq_option == 0:
        print("  -> 使用默认量化策略 (通常为全INT8)")
    elif args.ptq_option == 1:
        print("  -> 使用 NoClip 校准, 权重类型: int16 (高精度策略)")
        ptq_options.calibrate_method = 'NoClip'
        ptq_options.w_quant_type = 'int16'
    elif args.ptq_option == 2:
        print("  -> 使用 NoClip 校准, 整体类型: int16 (极高精度策略)")
        ptq_options.calibrate_method = 'NoClip'
        ptq_options.quant_type = 'int16'
    elif args.ptq_option == 4:
         print("  -> 使用高阶压缩量化策略 (Option 4)")

    ptq_options.set_tensor_data(generate_data(input_shape, ptq_options.samples_count, args.dataset))
    compiler.use_ptq(ptq_options)

    print("\n[STEP 5] 开始编译 KModel (这可能需要 1-5 分钟，请耐心等待)...")
    compile_start = time.time()
    compiler.compile()
    print(f"  -> 编译完成！耗时: {time.time() - compile_start:.2f} 秒")

    print("\n[STEP 6] 导出与清理...")
    kmodel = compiler.gencode_tobytes()
    base,ext=os.path.splitext(args.model)
    kmodel_name=base+".kmodel"
    with open(kmodel_name, 'wb') as f:
        f.write(kmodel)
    print(f"  -> 固件保存成功: {kmodel_name}")

    if os.path.exists("./tmp"):
        shutil.rmtree("./tmp")
    if os.path.exists("./gmodel_dump_dir"):
        shutil.rmtree("./gmodel_dump_dir")
        
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"✅ 全部流程顺利结束！总耗时: {total_time:.2f} 秒")
    print("="*60)

if __name__ == '__main__':
    main()