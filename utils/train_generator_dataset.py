import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np

class ImageEditDataset(Dataset):
    def __init__(self, all_images, org_prompts, all_targets, all_mask = None, device="cpu"):
        self.all_images = all_images
        self.org_prompts = org_prompts
        self.all_targets = all_targets
        self.num_images = all_images.shape[0]
        self.num_prompts = len(org_prompts)
        self.device = device
        self.all_mask = all_mask

    def __len__(self):
        return self.num_images * self.num_prompts

    def __getitem__(self, idx):
        img_idx = idx // self.num_prompts
        prompt_idx = idx % self.num_prompts
        print(f"Processing image {img_idx} with prompt {prompt_idx} ")

        image = self.all_images[img_idx, :, :, :].to(self.device)
        prompt = self.org_prompts[prompt_idx]
        target = self.all_targets[img_idx][prompt_idx].to(self.device)

        if self.all_mask is not None:
            mask = self.all_mask[img_idx, :, :, :].to(self.device)
            return image, prompt, mask, target, idx
        return image, prompt, target, idx
