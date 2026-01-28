import os
import sys

sys.path.append(os.getcwd())
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# 自定义的数据增强
def get_test_transform():
    return transforms.Compose(
        [
            transforms.Resize([640, 640]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# 推理的图片路径
image = Image.open("D:/ultralytics-main/ultralytics-main/datasets/handler/images/train/hander1_1.jpg")  # .convert('RGB')

img = get_test_transform()(image)
img = img.unsqueeze_(0)  # -> NCHW, 1,3,224,224
# 模型加载
onnx_model_path = r"D:/ultralytics-main/ultralytics-main/runs/detect/train12/weights/best.onnx"
resnet_session = onnxruntime.InferenceSession(onnx_model_path)
inputs = {resnet_session.get_inputs()[0].name: to_numpy(img)}
outs = resnet_session.run(None, inputs)[0]

print("onnx weights", outs)
print("onnx prediction", outs.argmax(axis=1)[0])
