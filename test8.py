from ultralytics import YOLO


if __name__ == '__main__':

    # 加载预训练的YOLOv11n模型
    model = YOLO(r"runs/detect/train12/weights/best.pt")

    # Run batched inference on a list of images
    results = model(["D:/ultralytics-main/ultralytics-main/datasets/hander/images/train/hander1_1.jpg"])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk