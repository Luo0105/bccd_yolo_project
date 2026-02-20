from ultralytics import YOLO
import os

# 确保路径正确
# 注意：YOLO 有时对相对路径支持有点迷，使用绝对路径最稳妥
current_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(current_dir, 'datasets', 'BCCD', 'data.yaml')

def train_yolo():
    # 1. 加载模型
    # 我们使用 yolov8n.pt (Nano版本)，它是最小最快的。
    # 对于BCCD这种简单任务，Nano足够了，而且训练飞快。
    # 如果你想追求更高精度，可以改用 'yolov8s.pt' (Small) 或 'yolov8m.pt' (Medium)
    print("🚀 Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt') 

    # 2. 开始训练
    # 这里的参数是针对 4060 显卡优化的
    print(f"🐢 Starting training on {data_yaml_path}...")
    
    results = model.train(
        data=data_yaml_path,   # 数据集配置文件路径
        epochs=50,             # 训练轮数 (50轮对于这个小数据集足够收敛了)
        imgsz=640,             # 图片大小 (标准YOLO输入)
        batch=16,              # 批次大小 (8GB显存开16-32都没问题，保守点开16)
        device=0,              # 使用 GPU 0
        workers=4,             # 数据加载线程数
        project='BCCD_Project',# 项目名称 (会生成在 runs/detect/BCCD_Project)
        name='yolov8n_run1',   # 本次实验名称
        exist_ok=True,         # 如果存在同名文件夹，覆盖它
        pretrained=True,       # 使用预训练权重 (迁移学习)
        optimizer='auto',      # 自动选择优化器 (通常是 SGD 或 AdamW)
        verbose=True           # 打印详细日志
    )

    print("✅ Training finished!")
    
    # 3. 验证模型 (在验证集上评估)
    metrics = model.val()
    print(f"Mean Average Precision @.5:.95 : {metrics.box.map}")

if __name__ == '__main__':
    # Windows 下多进程运行必须放在 if __name__ == '__main__': 之下
    train_yolo()