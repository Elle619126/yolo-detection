from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
import onnx
from ultralytics import YOLO
import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod, CalibrationDataReader

if __name__ == '__main__':

    model = YOLO('runs/detect/train12/weights/best.pt')  # 替换为你的模型路径
    # 1. 首先导出标准 ONNX 模型
    model.export(format='tflite', imgsz=640, int8=True)

    # 2. 加载并检查模型
    # onnx_model = onnx.load('runs/detect/train12/weights/best.onnx')
    # onnx.checker.check_model(onnx_model)

    #3. 进行动态量化
    # quantize_dynamic(
    #     'runs/detect/train12/weights/best.onnx',
    #     'runs/detect/train12/weights/best_int8.onnx',
    #     weight_type=QuantType.QUInt8,
    #    # optimize_model=True
    # )

    # quantize_static(
    #     'runs/detect/train6/weights/yolo11n.onnx',
    #     'runs/detect/train6/weights/yolo11n_int8.onnx',
    #     weight_type=QuantType.QUInt8,
    #    # optimize_model=True
    # )
    print("INT8 量化完成！")