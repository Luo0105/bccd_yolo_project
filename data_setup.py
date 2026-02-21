import os
import glob
import shutil
import xml.etree.ElementTree as ET
import random
from pathlib import Path
import stat

# configurations
ORIGINAL_REPO_URL = "https://github.com/Shenggan/BCCD_Dataset.git"

TEMP_DIR = "temp_download"

DATASET_DIR = "datasets/BCCD"

CLASSES = ['RBC', 'WBC', 'Platelets']

def convert_annotation(size, box):
    """coordinate conversion for YOLO format"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def setup_data():
    # 1. clean uo old data
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    if os.path.exists(DATASET_DIR):
        print(f"Warning: {DATASET_DIR} exists. It will be overwritten.")
        shutil.rmtree(DATASET_DIR)
    
    # 2. download original dataset
    print(f"🚀 Cloning original dataset from {ORIGINAL_REPO_URL}...")
    os.system(f"git clone {ORIGINAL_REPO_URL} {TEMP_DIR}")

    # 3. create dataset structure
    # datasets/BCCD/
    # ├── images/ (train, val)
    # └── labels/ (train, val)
    for split in ['train', 'val']:
        os.makedirs(f"{DATASET_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_DIR}/labels/{split}", exist_ok=True)

    # 4. get all xml files and process them
    xml_files = glob.glob(os.path.join(TEMP_DIR, "BCCD", "Annotations", "*.xml"))
    print(f"📂 Found {len(xml_files)} xml files. Processing...")

    # 5. shuffle and split dataset (80% train, 20% val)
    random.seed(42)
    random.shuffle(xml_files)
    
    split_idx = int(len(xml_files) * 0.8)
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]

    def process_files(files, split_name):
        for xml_file in files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            if w == 0 or h == 0: continue

            file_id = os.path.splitext(os.path.basename(xml_file))[0]
            img_src = os.path.join(TEMP_DIR, "BCCD", "JPEGImages", file_id + ".jpg")
            
            if not os.path.exists(img_src):
                img_src = os.path.join(TEMP_DIR, "BCCD", "JPEGImages", file_id + ".jpeg")
                if not os.path.exists(img_src):
                    continue

            label_txt_path = os.path.join(DATASET_DIR, "labels", split_name, file_id + ".txt")
            with open(label_txt_path, 'w') as out_file:
                for obj in root.iter('object'):
                    difficult = obj.find('difficult')
                    if difficult is not None and int(difficult.text) == 1:
                        continue
                    cls = obj.find('name').text
                    if cls not in CLASSES:
                        continue
                    cls_id = CLASSES.index(cls)
                    
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert_annotation((w, h), b)
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

            shutil.copy(img_src, os.path.join(DATASET_DIR, "images", split_name, file_id + ".jpg"))

    process_files(train_files, 'train')
    process_files(val_files, 'val')

    yaml_content = f"""
path: .
train: images/train
val: images/val

names:
  0: RBC
  1: WBC
  2: Platelets
    """
    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    # clean up temporary directory
    shutil.rmtree(TEMP_DIR, onerror=remove_readonly)
    
    print(f"Data setup complete. Dataset is ready at {DATASET_DIR}")
    print(f"'data.yaml' created at {os.path.join(DATASET_DIR, 'data.yaml')}")

if __name__ == "__main__":
    setup_data()