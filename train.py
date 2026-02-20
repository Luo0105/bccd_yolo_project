from ultralytics import YOLO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(current_dir, 'datasets', 'BCCD', 'data.yaml')

def train_yolo():
    # 1. load the model
    print("Loading model...")
    model = YOLO('yolov8n.pt') # for now we use the nano model, which is super lightweight and fast for small datasets

    # 2. train the model
    # this is for laptop 4060
    print(f"🐢 Starting training on {data_yaml_path}...")
    
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project='BCCD_Project',
        name='yolov8n_run1',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True
    )

    print("Training finished. Evaluating on validation set...")
    
    # 3. validate the model
    metrics = model.val()
    print(f"Mean Average Precision @.5:.95 : {metrics.box.map}")

if __name__ == '__main__':
    train_yolo()