import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging
from datetime import datetime

from torchvision.transforms import transforms

# 配置日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()  # 同时在控制台输出
    ]
)

logger = logging.getLogger(__name__)

class MyData(Dataset):

    def __init__(self,file_dir,transform=None):
        self.file_dir=file_dir
        self.transform=transform
        self.image_name,self.image_label=self.operate_file()

    def __len__(self):

        return len(self.image_name)

    def __getitem__(self,index):
        image=Image.open(self.image_name[index])
        image_label=self.image_label[index]
        if self.transform:
            image=self.transform(image)
        return image,image_label

    def operate_file(self):
        image_name=[]
        image_label=[]
        for filename in os.listdir(self.file_dir):
            if filename.endswith(".png"):
                image_name.append(os.path.join(self.file_dir,filename))
                image_label.append(int(filename.split("_")[0]))
        return image_name,image_label


def compute_mean_std(file_dir):
    sum_ = np.zeros(3)
    sum_sq = np.zeros(3)
    num_pixels = 0
    image_count = 0

    for filename in os.listdir(file_dir):
        if filename.endswith(".png"):
            try:
                image_path = os.path.join(file_dir, filename)
                img = Image.open(image_path).convert('RGB')
                img = np.array(img, dtype=np.float32) / 255.0  # 归一化并转换为float32
                img = img.transpose(2, 0, 1)  # 转为CHW格式

                num_pixels += img.shape[1] * img.shape[2]
                sum_ += img.sum(axis=(1, 2))
                sum_sq += (img ** 2).sum(axis=(1, 2))
                image_count += 1
            except Exception as e:
                logger.warning(f"Error processing {filename}: {str(e)}")
                continue

    mean = sum_ / num_pixels
    std = np.sqrt(sum_sq / num_pixels - mean ** 2)

    logger.info(f"Processed {image_count} images")
    logger.info(f"Computed mean: {mean}, std: {std}")
    return mean, std

def transform_and_dataloader(train_dir,test_dir,batch_size):
    mean, std = compute_mean_std(train_dir)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),  # 统一缩放到224x224
            transforms.RandomHorizontalFlip(),  # 数据增强
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    train_dataset = MyData(train_dir, data_transforms['train'])
    test_dataset = MyData(test_dir, data_transforms['val'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(test_dataset)}")
    return train_loader, val_loader
