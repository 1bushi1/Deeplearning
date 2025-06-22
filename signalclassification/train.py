import logging
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
from datetime import datetime
from torchvision.models.quantization import resnet18
from tqdm import tqdm
from data.MyData import transform_and_dataloader

# 设置随机种子保证可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 配置日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, num_epochs=25, device='cuda', patience=5):
    """完整的模型训练函数"""
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    start_time = time.time()

    # 将模型移到设备
    model = model.to(device)

    # 打印设备信息
    logger.info(f"Training on {device}")
    if str(device) == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    best_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        logger.info('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1} Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.cpu().numpy())
        logger.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1} Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc.cpu().numpy())
        logger.info(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logger.info(f'EarlyStop Counter: {early_stop_counter}/{patience}')

        if early_stop_counter >= patience:
            logger.info(f'Early stopping triggered at epoch {epoch + 1}!')
            logger.info(f'Best validation accuracy: {best_acc:.4f} at epoch {best_epoch + 1}')
            break

    time_elapsed = time.time() - start_time
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_acc:.4f}')
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    return model


def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plot_path = os.path.join(save_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Metrics plot saved at: {plot_path}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    train_dir = r"D:\datasets\RoadSignal\tsrd-train"
    test_dir = r"D:\datasets\RoadSignal\TSRD-Test"
    batch_size = 32

    train_loader, val_loader = transform_and_dataloader(train_dir, test_dir, batch_size)

    from models.MyResNet import MyResNet
    from models.MyVGG import MyVGG
    from models.MyDenseNet import DenseNet

    resnetmodel = MyResNet(18, num_classes=58)
    vggmodel = MyVGG(58)
    densemodel = DenseNet(121, growth_rate=32, num_classes=58)

    logger.info("Starting training on GPU..." if torch.cuda.is_available() else "Starting training on CPU...")

    # 开始训练
    trained_model1 = train_model(vggmodel, train_loader, val_loader, num_epochs=30, device=device)
    trained_model2 = train_model(resnetmodel, train_loader, val_loader, num_epochs=30, device=device)
    trained_model3 = train_model(densemodel, train_loader, val_loader, num_epochs=30, device=device)

    # 保存最终模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path1 = os.path.join("saved_models", f'final_vggmodel_{timestamp}.pth')
    final_model_path2 = os.path.join("saved_models", f'final_resnetmodel_{timestamp}.pth')
    final_model_path3 = os.path.join("saved_models", f'final_densemodel_{timestamp}.pth')

    torch.save(trained_model1.state_dict(), final_model_path1)
    torch.save(trained_model2.state_dict(), final_model_path2)
    torch.save(trained_model3.state_dict(), final_model_path3)

    logger.info(f'最终模型保存路径: {final_model_path1}')
    logger.info(f'最终模型保存路径: {final_model_path2}')
    logger.info(f'最终模型保存路径: {final_model_path3}')