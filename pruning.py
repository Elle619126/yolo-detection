from ultralytics import YOLO

model = YOLO('runs/detect/yolo11/weights/best.pt')
model = YOLO("yolo11n.pt")

def prunetrain(train_epochs, prune_epochs=0, quick_pruning=True, prune_ratio=0.5,
               prune_iterative_steps=1, data='coco.yaml', name='yolo11', imgsz=640,
               batch=8, device=[0], sparse_training=False):
    if not quick_pruning:
        print("Phase 1: Pre-training...")
        model.train(data=data, epochs=train_epochs, prune=False, sparse_training=sparse_training)

        print("Phase 2: Pruning...")
        best_weights = f"runs/detect/{name}_phase1/weights/best.pt"
        pruned_model = YOLO(best_weights)
        return pruned_model.train(data=data, epochs=prune_epochs, prune=True, prune_ratio=prune_ratio)
    else:
        return model.train(data=data, epochs=train_epochs, prune=True, prune_ratio=prune_ratio)


if __name__ == '__main__':
    prunetrain(
        train_epochs=10,
        prune_epochs=20,
        quick_pruning=False,
        prune_ratio=0.5,
        prune_iterative_steps=1,
        data='uno.yaml',
        batch=8,
        imgsz=640,
        device=[0],
        name='yolo11_prune',
        sparse_training=True
    )
