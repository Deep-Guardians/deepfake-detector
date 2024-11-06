import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data.deepfake_dataset import DeepfakeDataset
from models.inception_resnet_v2 import InceptionResNetV2
from utils.logger import get_logger
from torchvision import transforms
import wandb
from tqdm import tqdm
import os
import yaml
import argparse
import heapq


# Configuration loading
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train_model(model, criterion, optimizer, dataloader, device, logger):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device).float()
        labels = labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        wandb.log({"train_loss": loss.item()})
    epoch_loss = running_loss / len(dataloader.dataset)
    logger.info(f"Training Loss: {epoch_loss:.4f}")
    return epoch_loss


def validate_model(model, criterion, dataloader, device, logger):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device).float()
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            wandb.log({"val_loss": loss.item()})
    epoch_loss = running_loss / len(dataloader.dataset)
    logger.info(f"Validation Loss: {epoch_loss:.4f}")
    return epoch_loss


def save_checkpoint(model, epoch, val_loss, checkpoint_dir="models/checkpoints", max_checkpoints=10):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save the model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"deepfake_detector_epoch{epoch}_val_loss{val_loss:.4f}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")

    # Get list of all checkpoints
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

    # If the number of checkpoints exceeds the maximum, remove the oldest checkpoint
    if len(checkpoints) > max_checkpoints:
        # Sort checkpoints by creation time
        checkpoints.sort(key=os.path.getctime)
        oldest_checkpoint = checkpoints[0]
        os.remove(oldest_checkpoint)
        print(f"Removed oldest checkpoint: {oldest_checkpoint}")


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="deepfake_detection")

    # Argument parsing
    parser = argparse.ArgumentParser(description="멀티모달 딥페이크 탐지 모델 학습 스크립트")
    parser.add_argument("--epochs", type=int, default=10, help="학습 반복 횟수 (기본값: 10)")
    parser.add_argument("--batch_size", type=int, default=4, help="훈련 배치 크기 (기본값: 4)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="학습률 (기본값: 0.001)")
    parser.add_argument("--real_dir", type=str, default="./data/REAL", help="실제 이미지가 저장된 디렉토리 경로 (기본값: ./data/REAL)")
    parser.add_argument("--fake_dir", type=str, default="./data/FAKE", help="가짜 이미지가 저장된 디렉토리 경로 (기본값: ./data/FAKE)")
    parser.add_argument("--gpu", type=int, default=0, help="학습에 사용할 GPU ID (기본값: 0)")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override configuration with command line arguments if provided
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["train"]["learning_rate"] = args.learning_rate
    if args.real_dir:
        config["data"]["real_dir"] = args.real_dir
    if args.fake_dir:
        config["data"]["fake_dir"] = args.fake_dir

    # Logger setup
    logger = get_logger("Train")

    # Data loading
    real_dir = config["data"]["real_dir"]
    fake_dir = config["data"]["fake_dir"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = DeepfakeDataset(real_dir, fake_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be performed on CPU.")
        device = torch.device("cpu")
    else:
        logger.info("CUDA is available. Training on GPU.")

    model = InceptionResNetV2().to(device)
    print(f"Using device: {device}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    # Training loop
    num_epochs = config["train"]["epochs"]
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train_model(model, criterion, optimizer, train_loader, device, logger)
        val_loss = validate_model(model, criterion, val_loader, device, logger)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss_epoch": train_loss, "val_loss_epoch": val_loss})

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, epoch + 1, val_loss)

    # Finalizing
    wandb.finish()
