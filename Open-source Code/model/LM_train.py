import os
from PIL import Image
import numpy as np
import clip
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

class dataprocess(Dataset):
    def __init__(self, img_root, is_train, preprocess):
        self.img_root = img_root
        self.is_train = is_train
        self.img_process = preprocess
        self.samples = []
        self.sam_labels = []
        self.rotate = transforms.RandomRotation(15)
        self.flip = transforms.RandomHorizontalFlip()
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.2, hue=0.4)
        self.augmentation = transforms.Compose([self.rotate, self.flip, self.color_jitter, preprocess])
        self.img_paths, self.sam_labels = self._get_image_paths(img_root)
        self.tokens = clip.tokenize(self.sam_labels)

    def _get_image_paths(self, root_dir):
        img_paths = []
        sam_labels = []
        for label in ['Tumor', 'Tissue', 'Background', 'ROI']:
            category_path = os.path.join(root_dir, label)
            for subdir in tqdm(os.listdir(category_path), desc=f"Reading {label} images"):
                subdir_path = os.path.join(category_path, subdir)
                for img_file in os.listdir(subdir_path):
                    if img_file.endswith('.png' or '.jpg'):
                        img_paths.append(os.path.join(subdir_path, img_file))
                        sam_labels.append(label + ' image of the liver.')
        return img_paths, sam_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        token = self.tokens[idx]
        image = Image.open(img_path).convert('RGB')
        if self.is_train:
            image = self.augmentation(image)
        else:
            image = self.img_process(image)
        return image, token

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    net.load_state_dict(torch.load("path_to_pre-trained_model.pth"))

    optimizer = optim.Adam(net.parameters(), lr=1e-5, betas=(0.9,0.98), eps=1e-6, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    your_dataset = dataprocess(img_root='path_to_train_dataset',
                               is_train=True, preprocess=preprocess)
    dataset_size_your = len(your_dataset)
    your_dataloader = DataLoader(your_dataset, batch_size=256, shuffle=True, num_workers=12, pin_memory=True)

    phase = "train"
    model_name = "ViT-B-32"
    ckt_gap = 5
    batch_num = 0
    for epoch in range(0, epoches):
        scheduler.step()
        total_loss = 0
        batch_num += 1
        with torch.cuda.amp.autocast(enabled=True):
            for images, label_tokens in tqdm(your_dataloader, desc="Processing batches"):
                images = images.to(device)
                label_tokens = label_tokens.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits_per_image, logits_per_text = net(images, label_tokens)
                    ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                    cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                    total_loss += cur_loss
                    if phase == "train":
                        cur_loss.backward()
                        if device == "cpu":
                            optimizer.step()
                        else:
                            optimizer.step()
                            clip.model.convert_weights(net)
                if batch_num % 4 == 0:
                    logger.info('{} epoch:{} loss:{}'.format(phase,epoch,cur_loss))
            epoch_loss = total_loss / dataset_size_your
            if epoch % ckt_gap == 0:
                torch.save(net.state_dict(),f"{model_name}_epoch_{epoch}.pth")
            logger.info(f"weights_{epoch} saved")
            if epoch % ckt_gap == 0:
                checkpoint_path = f"{model_name}_ckt.pth"
                checkpoint = {
                    'it': epoch,
                    'network': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")
            logger.info('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

if __name__ == '__main__':
    epoches = 51
    main()