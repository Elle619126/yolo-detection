from ultralytics import YOLO


if __name__ == '__main__':

    # 加载你的 YOLOv11 模型
    model = YOLO('runs/detect/train6/weights/best.pt')  # 替换为你的模型路径
    model.train( data="ultralytics/cfg/datasets/handler.yaml",
                epochs=1, prune=True, prune_ratio=0.5)
    # 导出为 ONNX
    #model.export(format='tflite',int8=True)

    #model.export(format='onnx', int8=True)