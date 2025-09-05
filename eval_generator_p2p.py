"""make variations of input image"""

import argparse, os
import copy


import torch
import sys
import PIL
import matplotlib.pyplot as plt
from fractions import Fraction
from pytorch_lightning import seed_everything
from PIL import Image
from diffusion.diff_process import diff_process
import pandas as pd
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
import time
import wandb
import re

from datetime import timedelta
from utils.test_detector import nude_test, unsafe_test
from utils.train_generator_dataset import ImageEditDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.util import *
from diffusion.safety_checker import *
from diffusion.DDPM_process import DDPM_process

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
    vae, tokenizer, text_encoder, unet, scheduler, pipe = get_sd_model(args, args.version)
    pipe = pipe.to(device, dtype=dtype)
    device = pipe.device

    num_images_per_prompt = 1
    args.eps = float(Fraction(args.eps))
    scaled_eps = round(args.eps * 255)

    num_images = 200
    num_prompt = 10
    num_unseen_img = 50
    num_sample = num_images * num_prompt

    split = int(9/10 * num_sample)

    work_path = f"your_path/adv_diffusion/"
    input_path = f"{work_path}dataset/naked_imgs/"


    output_path = (f"{work_path}hf_results/{args.prefix}sd{args.version}_{args.concept}_{args.loss_type}_num_{num_sample}"
                   f"_vec_{args.vec_scale}_step{args.tar_steps}/")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}noise", exist_ok=True)
    img_save_path = f"{output_path}imgs_{args.noi_type}_eps{scaled_eps}_lr{args.lr}/"
    input_img_save_path = f"{output_path}input_imgs_{args.noi_type}_eps{scaled_eps}_lr{args.lr}/"
    unseen_img_save_path = f"{output_path}unseen_imgs_{args.noi_type}_eps{scaled_eps}_lr{args.lr}/"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(input_img_save_path, exist_ok=True)
    os.makedirs(unseen_img_save_path, exist_ok=True)
    clear_directory(img_save_path)
    clear_directory(input_img_save_path)
    clear_directory(unseen_img_save_path)

    args_dict = vars(args)
    wandb.init(
        # set the wandb project where this run will be logged
        entity="your_wandb_account",
        project="HF_AdvDiffusion",
        name=f"eval_{args.concept}_{args.version}_{args.prefix}{num_sample}samples_lr{args.lr}_{args.loss_type}"
              f"_tarstep{args.tar_steps}_vec{args.vec_scale}_eps{scaled_eps}",
        # track hyperparameters and run metadata
        config=args_dict
    )

    if args.attack == "vae":
        target_path = f"your_path/adv_diffusion/hf_results/vaesdp2p_nudity_mse_num_2000_vec_2.5_step8/2.5.pt"
    else:
        target_path = f"your_path/adv_diffusion/hf_results/basesdp2p_nudity_mse_num_2000_vec_2.5_step8/2.5.pt"

    args_dict = vars(args)


    precision_scope = nullcontext

    if args.defense == "sc" or "sc" in args.attack:
        surrogate_sc = StableDiffusionSafetyChecker(pipe.safety_checker.config, pipe.safety_checker).to(device)
        sc_loss_factor = 0.2

    imgs = os.listdir(input_path)
    with torch.no_grad():
        all_images = [load_and_preprocess_image(os.path.join(input_path, i), pipe=pipe,
                                                resolution=args.resolution, device=device, dtype=dtype) for i in imgs]
        all_images = torch.cat(all_images, dim=0).cpu()
        test_images = all_images[num_images: min(num_images + num_unseen_img, len(all_images))]
        train_imgs = all_images[:num_images]
        use_imgs = torch.cat([train_imgs,test_images], dim=0)

        if args.attack == "MMA":
            org_prompts = get_orig_prompts(num_prompt, concept=args.concept)
        else:
            org_prompts = get_orig_prompts(num_prompt, concept=args.concept)
        org_prompts = [""] * len(org_prompts)
        all_targets = torch.load(target_path)



    dataset = ImageEditDataset(train_imgs, org_prompts, all_targets, device=device)
    indices = list(range(num_sample))
    np.random.shuffle(indices)
    train_indices, eval_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    eval_sampler = SubsetRandomSampler(eval_indices)

    train_loader = DataLoader(dataset, batch_size=args.bs, sampler=train_sampler, drop_last=False)
    eval_loader = DataLoader(dataset, batch_size=args.bs, sampler=eval_sampler, drop_last=False)

    torch_device = pipe.device

    if args.noise_gen == "adv_noise":
        noise_generator = torch.randn_like(all_images[0], requires_grad=True) * 2 * args.eps - args.eps
        noise_generator = noise_generator.to(device)
        noise_generator = noise_generator.detach().requires_grad_()
        model_para = [noise_generator]
    else:
        noise_generator, _, _, _, _, _ = get_sd_model(args, "2-1")
        if args.ckpt is not None and os.path.exists(args.ckpt):
            ckpt = torch.load(args.ckpt)
            noise_generator.load_state_dict(ckpt)

        noise_generator = noise_generator.to(device)
        for param in noise_generator.parameters():
            param.requires_grad = True
        model_para = noise_generator.parameters()

    adam_eps = 1e-5 if args.dtype == "float16" else 1e-8
    optimizer = optim.AdamW(model_para, lr=args.lr, eps=adam_eps)


    if "sld" in args.defense:
        _, _, _, _, _, sld_pipe = get_sd_model(args, "1-5-safe")
    else:
        sld_pipe = None

    print("Start testing.")
    if args.noise_gen != "adv_noise":
        noise_generator.eval()
    with torch.no_grad():
        for iter, (images, prompts, targets, sample_i) in enumerate(eval_loader):
            if "3-0" in args.version:
                noisy_imgs = noise_generator(images).sample
                if args.noi_type == "constrain":
                    constrained_noisy_imgs = torch.min(torch.max(noisy_imgs, images - args.eps),
                                                       images + args.eps)
                else:
                    constrained_noisy_imgs = noisy_imgs
                prompts = list(prompts)

                imgs = pipe(prompt=prompts, image=constrained_noisy_imgs, height=512, width=512, num_inference_steps=8,
                     guidance_scale=7.5, strength=0.8).images
                eval_loss = 0
                do_denormalize = [True] * len(imgs)

            else:
                with precision_scope():
                    # test noise
                    prompts = list(prompts)
                    if args.defense == "ng":
                        if args.concept == "nudity":
                            negative_prompt = ["nudity"]
                        elif args.concept == "violence":
                            negative_prompt = ["violence"]
                        else:
                            negative_prompt = [""]
                        negative_prompt = negative_prompt * args.bs
                    else:
                        negative_prompt = None
                    if args.version == "p2p":
                        prompt_embedding = pipe._encode_prompt(prompts, device=device,
                                                               num_images_per_prompt=num_images_per_prompt,
                                                               negative_prompt=negative_prompt,
                                                               do_classifier_free_guidance=True)
                        negative_prompt_embeds = None
                    else:
                        prompt_embedding, negative_prompt_embeds = pipe.encode_prompt(prompts, device=device,
                                                                                      num_images_per_prompt=num_images_per_prompt,
                                                                                      negative_prompt=negative_prompt,
                                                                                      do_classifier_free_guidance=True)

                    if args.loss_type == "cos":
                        base_z_p, base_noise_pred, base_noise_pred_list = diff_process(pipe, copy.deepcopy(scheduler), args, images,
                                                                 prompt_embedding, negative_prompt_embeds, torch_device,
                                                                 diff_noise_gen, args.bs, num_images_per_prompt,
                                                                 defense=args.defense, get_target=False, sld_pipe=sld_pipe)
                    if args.noise_gen == "adv_noise":
                        noisy_imgs = noise_generator + images
                    elif args.noise_gen != "vae":
                        noisy_imgs = noise_generator(images)
                    else:
                        noisy_imgs = noise_generator(images).sample

                    if args.attack == "MMA":
                        noisy_imgs = images

                    if args.defense == "noise":
                        noisy_imgs = noisy_imgs + torch.clamp(torch.randn_like(noisy_imgs[0]), max=args.eps,
                                                              min=-args.eps)

                    if args.noi_type == "constrain":
                        constrained_noisy_imgs = torch.min(torch.max(noisy_imgs, images - args.eps),
                                                           images + args.eps)
                    else:
                        constrained_noisy_imgs = noisy_imgs

                    z_p, noise_pred, noise_pred_list = diff_process(pipe, copy.deepcopy(scheduler), args, constrained_noisy_imgs,
                                                   prompt_embedding, negative_prompt_embeds, torch_device, diff_noise_gen,
                                                   args.bs, num_images_per_prompt, defense=args.defense, get_target=False, sld_pipe=sld_pipe)

                diff_img = pipe.vae.decode(z_p / pipe.vae.config.scaling_factor, return_dict=False,
                                           generator=diff_noise_gen)[0]
                if args.defense == "sc":
                    diff_img, has_nsfw_concept = pipe.run_safety_checker(diff_img, device, dtype)

                do_denormalize = [True] * diff_img.shape[0]
                imgs = pipe.image_processor.postprocess(diff_img.detach().cpu(), output_type="pil",
                                                       do_denormalize=do_denormalize)
                for idx, i in enumerate(imgs):
                    img_idx = sample_i[idx] // num_prompt
                    prompt_idx = sample_i[idx] % num_prompt
                    i.save(f"{img_save_path}{args.noise_gen}_eval_img{img_idx}_p{prompt_idx}.jpg")

            constrained_noisy_imgs = pipe.image_processor.postprocess(constrained_noisy_imgs.detach().cpu(),
                                                                     output_type="pil", do_denormalize=do_denormalize)
            for idx, i in enumerate(constrained_noisy_imgs):
                img_idx = sample_i[idx] // num_prompt
                prompt_idx = sample_i[idx] % num_prompt
                i.save(f"{input_img_save_path}{args.noise_gen}_input_img{img_idx}_p{prompt_idx}.jpg")

            if iter >= 24:
                break

        if args.concept == "nudity":
            ASR, failed_sample = nude_test(data_path=f"{img_save_path}")
        elif args.concept == "violence":
            ASR, failed_sample = unsafe_test(data_path=f"{img_save_path}", device=device)
    wandb.log({"ASR": ASR})

    print("ASR:", ASR)
    print(f"Your samples are ready and waiting for you here: \n{output_path} \nEnjoy.")


if __name__ == "__main__":
    main()
