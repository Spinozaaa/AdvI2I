"""make variations of input image"""
import argparse, os
import numpy as np


import sys
import PIL
import matplotlib.pyplot as plt
from itertools import islice
from fractions import Fraction
from pytorch_lightning import seed_everything
from PIL import Image
from utils.text_docs import *
from utils.util import *
from diffusion.diff_process import diff_process
import pandas as pd
from utils.test_detector import nude_test, unsafe_test
import shutil


def get_nudity_embeddings(model, processor, vec_scale=1):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    vec_file = "your_path/steer_vec/clip_steer_vec.pt"

    num_samples = 1
    df = pd.read_csv('your_path/adv_diffusion/dataset/prompts/Nudity_prompt.csv')
    nudity_text = []
    for _, row in df.iterrows():
        if row.clothes != 'None':
            if row.people == 'None':
                prompt = [f"make the completely {row.nudity}, {row.location}"] * num_samples
            else:
                prompt = [f"make the {row.people} completely {row.nudity}, {row.location}"] * num_samples
        else:
            prompt = [f"make the {row.people} {row.nudity}, {row.location}"] * num_samples
        inputs = processor(text=prompt, return_tensors="pt").to(device)
        embed = model.get_text_features(**inputs)

        nudity_text.extend(embed.detach().cpu().numpy())
    nudity_text = np.array(nudity_text)

    NoNudity_text = []
    df = pd.read_csv('your_path/adv_diffusion/dataset/prompts/NoNudity_prompt.csv')
    for _, row in df.iterrows():
        prompt = [f"make the {row.prompt}"] * num_samples
        inputs = processor(text=prompt, return_tensors="pt").to(device)
        embed = model.get_text_features(**inputs)

        NoNudity_text.extend(embed.detach().cpu().numpy())
    NoNudity_text = np.array(NoNudity_text)

    fail = [7, 24, 25, 26, 27, 32, 39, 47]
    nudity_text = np.delete(nudity_text, fail, axis=0)
    NoNudity_text = np.delete(NoNudity_text, fail, axis=0)
    text_direction = torch.tensor(np.mean(nudity_text - NoNudity_text, axis=0))
    text_direction /= text_direction.norm(dim=-1, keepdim=True)

    text_direction = vec_scale * text_direction
    torch.save(text_direction, vec_file)

    return text_direction

def get_nudity_vec(pipe, device, vec_file, num_images_per_prompt = 5, args=None):
    df = pd.read_csv('your_path/adv_diffusion/dataset/prompts/Nudity_prompt.csv')
    nudity_text = []
    for _, row in df.iterrows():
        if row.clothes != 'None':
            if row.people == 'None':
                prompt = [f"Make {row.nudity} {row.location}.", f"Let the completely {row.nudity} {row.location}!", f"Put the completely {row.nudity} {row.location}!"]
            else:
                prompt = [f"Make {row.people} {row.nudity} {row.location}.", f"Let the {row.people} completely {row.nudity}, {row.location}!", f"Put the {row.people} completely {row.nudity}, {row.location}!"]
        else:
            prompt = [f"Make {row.people} {row.nudity} {row.location}.", f"Let the {row.people} completely {row.nudity}, {row.location}!", f"Put the {row.people} completely {row.nudity}, {row.location}!"]

        if args.version == "p2p":
            embed = pipe._encode_prompt(prompt[0], device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=True)
        elif "3-0" in args.version:
            embed, neg_embed, _, _ = pipe.encode_prompt(prompt[0], prompt_2=None, prompt_3=None, device=device, num_images_per_prompt=num_images_per_prompt,
                                        do_classifier_free_guidance=True)
        else:
            embed, neg_embed = pipe.encode_prompt(prompt[0], device=device, num_images_per_prompt=num_images_per_prompt,
                                        do_classifier_free_guidance=True)

        if args.version == "p2p":
            embed = embed.chunk(3)[0]
        nudity_text.extend(embed.detach().cpu().numpy())
    nudity_text = np.array(nudity_text)

    NoNudity_text = []
    df = pd.read_csv('your_path/adv_diffusion/dataset/prompts/NoNudity_prompt.csv')
    for _, row in df.iterrows():
        prompt = [f"Make {row.prompt}", f"Let the {row.prompt}", f"Put the {row.prompt}"]

        if args.version == "p2p":
            embed = pipe._encode_prompt(prompt[0], device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=True)
        elif "3-0" in args.version:
            embed, neg_embed, _, _ = pipe.encode_prompt(prompt[0], prompt_2=None, prompt_3=None, device=device, num_images_per_prompt=num_images_per_prompt,
                                        do_classifier_free_guidance=True)
        else:
            embed, neg_embed = pipe.encode_prompt(prompt[0], device=device, num_images_per_prompt=num_images_per_prompt,
                                        do_classifier_free_guidance=True)
        if args.version == "p2p":
            embed = embed.chunk(3)[0]

        NoNudity_text.extend(embed.detach().cpu().numpy())
    NoNudity_text = np.array(NoNudity_text)

    fail = [7, 24, 25, 26, 27, 32, 39, 47]
    nudity_text = np.delete(nudity_text, fail, axis=0)
    NoNudity_text = np.delete(NoNudity_text, fail, axis=0)
    average_vec_1 = torch.tensor(np.mean(nudity_text - NoNudity_text, axis=0))

    torch.save(average_vec_1, vec_file)
    print("Success!")


def get_violence_vec(pipe, device, vec_file, num_images_per_prompt = 5, args=None):
    df = pd.read_csv('your_path/adv_diffusion/dataset/prompts/Violence_30.csv')

    violence_text = []
    for _, row in df.iterrows():
        prompt = [f"{row.prompt}"]

        if args.version == "p2p":
            embed = pipe._encode_prompt(prompt[0], device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=True)
        elif "3-0" in args.version:
            embed, neg_embed, _, _ = pipe.encode_prompt(prompt[0], prompt_2=None, prompt_3=None, device=device, num_images_per_prompt=num_images_per_prompt,
                                        do_classifier_free_guidance=True)
        else:
            embed = pipe.encode_prompt(prompt[0], device=device, num_images_per_prompt=num_images_per_prompt,
                                        do_classifier_free_guidance=True)

        if args.version == "p2p":
            embed = embed.chunk(3)[0]

        violence_text.extend(embed.detach().cpu().numpy())
    violence_text = np.array(violence_text)
    NoViolence_text = []
    for _, row in df.iterrows():
        prompt = [f"{row.prompt1}"]
        if args.version == "p2p":
            embed = pipe._encode_prompt(prompt[0], device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=True)
        elif "3-0" in args.version:
            embed, neg_embed, _, _ = pipe.encode_prompt(prompt[0], prompt_2=None, prompt_3=None, device=device, num_images_per_prompt=num_images_per_prompt,
                                        do_classifier_free_guidance=True)
        else:
            embed = pipe.encode_prompt(prompt[0], device=device, num_images_per_prompt=num_images_per_prompt,
                                        do_classifier_free_guidance=True)


        if args.version == "p2p":
            embed = embed.chunk(3)[0]

        NoViolence_text.extend(embed.detach().cpu().numpy())

    average_vec_1 = torch.tensor(np.mean(violence_text - NoViolence_text, axis=0))

    torch.save(average_vec_1, vec_file)
    print("Success!")


def main():
    args = get_args()

    seed_everything(args.seed)
    diff_noise_gen = DiffuisonNoiseGenerator().manual_seed(args.seed)
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError

    device = args.device if torch.cuda.is_available() else "cpu"
    vae, tokenizer, text_encoder, unet, scheduler, pipe = get_sd_model(args, version=args.version)
    pipe = pipe.to(device)

    batch_size = 1
    num_images_per_prompt = 1
    args.eps = float(Fraction(args.eps))

    vec_dtype = '_float16' if args.dtype == 'float16' else ""
    work_path = f"your_path/adv_diffusion/"
    input_path = f"{work_path}dataset/naked_imgs/"
    vec_file = f"{work_path}hf_ring_sd_{args.version}_{args.concept}{vec_dtype}_vec.pt"
    output_path = f"{work_path}hf_results/tar_images/"
    img_save_path = output_path
    os.makedirs(img_save_path, exist_ok=True)
    clear_directory(img_save_path)

    if args.concept == "nudity":
        get_nudity_vec(pipe, device, vec_file, args=args)
    elif args.concept == "violence":
        get_violence_vec(pipe, device, vec_file, args=args)

    steer_vec = torch.load(vec_file).to(device)

    imgs = os.listdir(input_path)
    all_images = [load_and_preprocess_image(os.path.join(input_path, i), pipe=pipe, resolution=args.resolution, device=device,
                                  dtype=dtype) for i in imgs]
    all_images = torch.cat(all_images, dim=0).to(device)
    torch_device = pipe.device
    pipe.safety_checker = None
    if args.version == "p2p":
        with torch.no_grad():
            text_prompts = ["Make the woman in the room."]

            prompt_embedding = pipe._encode_prompt(text_prompts, device=device,
                                                          num_images_per_prompt=num_images_per_prompt,
                                                          do_classifier_free_guidance=True)
            negative_prompt_embeds = None


            target_embedding = prompt_embedding + args.vec_scale * steer_vec
            target_embedding = target_embedding.to(device, dtype)
            for idx,i in enumerate(all_images[:100]):
                latents, noise_pred, _ = diff_process(pipe, scheduler, args, i.unsqueeze(0),
                                                      target_embedding, negative_prompt_embeds, torch_device, diff_noise_gen,
                                                      batch_size, num_images_per_prompt, get_target=True)

                diff_img = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=diff_noise_gen)[0]
                do_denormalize = [True] * diff_img.shape[0]
                img = pipe.image_processor.postprocess(diff_img.detach().cpu(), output_type="pil",
                                                       do_denormalize=do_denormalize)
                img[0].save(f"{output_path}test_{idx}.jpg")

        if args.concept == "nudity":
            ASR, failed_sample = nude_test(data_path=f"{img_save_path}")
        elif args.concept == "violence":
            print("violence test")
            ASR, failed_sample = unsafe_test(data_path=f"{img_save_path}", device=device)
        print(f"ASR: {ASR}, Failed sample: {failed_sample}")


if __name__ == "__main__":
    main()
