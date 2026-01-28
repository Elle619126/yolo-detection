import tflite_runtime.interpreter as tflite	# 改动一
import cv2
import numpy as np


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    tensor = np.expand_dims(image, axis=[0, -1])
    tensor = tensor.astype('float32')
    return tensor




emotion_model_tflite = tflite.Interpreter("output.tflite")	# 改动二

emotion_model_tflite.allocate_tensors()
tflife_input_details = emotion_model_tflite.get_input_details()
tflife_output_details = emotion_model_tflite.get_output_details()

img = cv2.imread("1fae49da5f2472cf260e3d0aa08d7e32.jpeg")
input_tensor = preprocess(img)


emotion_model_tflite.set_tensor(tflife_input_details[0]['index'], input_tensor)

emotion_model_tflite.invoke()

custom = emotion_model_tflite.get_tensor(tflife_output_details[0]['index'])

print(custom)

