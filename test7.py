import onnx
from onnxconverter_common import float16

if __name__ == "__main__":
    model = onnx.load("runs/detect/train6/weights/best.onnx")

    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, "runs/detect/train6/weights/best_fp16.onnx")
