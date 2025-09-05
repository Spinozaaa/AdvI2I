import os
from PIL import Image
import numpy as np
import torch
import cv2
from utils import detect, segment, draw_mask, generate_image, remove_special_chars, ask_chatgpt, ImageSimilarity
from GroundingDINO.groundingdino.util.inference import load_image, load_model

def process_images(input_dir, target_size=(512, 512)):
    image_sources = []
    input_images_name = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image_source, _ = load_image(image_path)
            h, w = image_source.shape[:2]

            if h < target_size[0] or w < target_size[1]:
                top = max((target_size[0] - h) // 2, 0)
                bottom = max(target_size[0] - h - top, 0)
                left = max((target_size[1] - w) // 2, 0)
                right = max(target_size[1] - w - left, 0)

                image_source = cv2.copyMakeBorder(image_source, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_source = cv2.resize(image_source, target_size)
            image_sources.append(image_source)

            print(f"Processed image: {filename}")
            input_images_name.append(filename)
    return image_sources, input_images_name

def init_sam(device):
    from segment_anything import SamPredictor, build_sam
    sam_checkpoint = './checkpoints/sam_vit_h_4b8939.pth'
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    return sam_predictor


def generate_mask(image_source, boxes_xyxy, sam_predictor):
    # image_source = torch.tensor(image_source)
    # print(image_source.shape)
    segmented_frame_masks = segment(image_source, sam_predictor, boxes_xyxy=boxes_xyxy, multimask_output=False, check_white=False)
    merged_mask = segmented_frame_masks[0]
    if len(segmented_frame_masks) > 1:
        for _mask in segmented_frame_masks[1:]:
            merged_mask = merged_mask | _mask
    return merged_mask

def save_mask_as_image(mask, image_path, output_dir):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    mask_image = Image.fromarray((mask.detach().cpu().numpy() * 255).astype(np.uint8))
    output_path = os.path.join(output_dir, f"{filename}_mask.png")
    mask_image.save(output_path)
    print(f"Mask saved as {output_path}")

work_path = f"/data/yaopei/adv_diffusion/"
input_path = f"{work_path}dataset/naked_imgs_easy/"
output_path = f"{work_path}dataset/naked_imgs_easy_masks/"
os.makedirs(output_path, exist_ok=True)

image_sources, input_images_name = process_images(input_path)
boxes_xyxy = torch.tensor([[50, 200, 500, 500]])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam_predictor = init_sam(device)

for img, img_name in zip(image_sources, input_images_name):
    mask = generate_mask(img, boxes_xyxy, sam_predictor)
    save_mask_as_image(mask, img_name, output_path)
