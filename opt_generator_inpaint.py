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
from datetime import timedelta
from utils.test_detector import nude_test, unsafe_test
from utils.train_generator_dataset import ImageEditDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.util import *
from diffusion.safety_checker import *
from diffusion.DDPM_process import DDPM_process

def main():

    args = get_args()
    print(args)

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
    indices = list(range(num_sample))
    np.random.shuffle(indices)
    split = int(9/10 * num_sample)
    train_indices, test_indices = indices[:split], indices[split:]
    test_bs = args.bs

    work_path = f"your_path/adv_diffusion/"
    input_path = f"{work_path}dataset/naked_imgs/"
    input_mask_path = f"{work_path}dataset/{args.mask_dir}/"

    vec_dtype = "_float16" if args.dtype == 'float16' else ""
    vec_file = f"{work_path}hf_ring_sd_{args.version}_{args.concept}{vec_dtype}_vec.pt"

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

    target_path = os.path.join(output_path, f"{args.vec_scale}.pt")

    args_dict = vars(args)
    wandb.init(
        # set the wandb project where this run will be logged
        entity="your_wandb_account",
        project="HF_AdvDiffusion",
        name=f"{args.concept}_{args.version}_{args.prefix}{num_sample}samples_lr{args.lr}_{args.loss_type}"
              f"_tarstep{args.tar_steps}_vec{args.vec_scale}_eps{scaled_eps}",
        # track hyperparameters and run metadata
        config=args_dict
    )

    steer_vec = torch.load(vec_file).to(device)
    precision_scope = nullcontext

    if args.defense == "sc" or "sc" in args.attack:
        surrogate_sc = StableDiffusionSafetyChecker(pipe.safety_checker.config, pipe.safety_checker).to(device)
        sc_loss_factor = 0.2

    with torch.no_grad():
        imgs_paths = os.listdir(input_path)
        masks_paths = os.listdir(input_mask_path)

        imgs_paths = sorted(imgs_paths, key=lambda x: extract_number_from_filename(x))
        masks_paths = sorted(masks_paths, key=lambda x: extract_number_from_filename(x))

        all_images = []
        all_masks = []
        all_masked_image = []
        with open(f'{output_path}output_file.txt', 'w') as f:
            for idx, img_name in enumerate(imgs_paths):
                f.write(f"{idx}: {img_name}\n")

        for img_path, mask_path in zip(imgs_paths, masks_paths):
            img_number = extract_number_from_filename(img_path)
            mask_number = extract_number_from_filename(mask_path)

            img_path = os.path.join(input_path, img_path)
            mask_path = os.path.join(input_mask_path, mask_path)

            if img_number != mask_number:
                print(f"Image {img_path} and Mask {mask_path} do not match!")
            else:
                image, processed_mask, masked_image = \
                    load_and_preprocess_image_mask(img_path, mask_path, pipe=pipe, dtype=dtype)
                all_images.append(image)
                all_masks.append(processed_mask)
                all_masked_image.append(masked_image)

        all_images = torch.cat(all_images, dim=0).cpu()

        train_imgs = all_images[:num_images]
        test_images = all_images[num_images: min(num_images+num_unseen_img, len(all_images))]
        use_imgs = torch.cat([train_imgs,test_images], dim=0)

        all_masks = torch.cat(all_masks, dim=0).cpu()
        train_masks = all_masks[:num_images]
        test_masks = all_masks[num_images: min(num_images+num_unseen_img, len(all_masks))]
        use_masks = torch.cat([train_masks, test_masks], dim=0)

        org_prompts = get_orig_prompts(num_prompt, concept=args.concept)

        get_target(use_imgs, org_prompts, pipe, args, steer_vec, precision_scope, device,
                   target_path, output_path, diff_process, diff_noise_gen,
                   copy.deepcopy(scheduler), masks=use_masks)
        all_targets = torch.load(target_path)

    with torch.no_grad():
        if args.concept == "nudity":
            ASR, failed_sample = nude_test(data_path=f"{output_path}target/")
        elif args.concept == "violence":
            ASR, failed_sample = unsafe_test(data_path=f"{output_path}target/", device=device)
            print("Get targets ASR:", ASR.data, failed_sample)
    torch.cuda.empty_cache()


    dataset = ImageEditDataset(train_imgs, org_prompts, all_targets, all_mask=train_masks, device=device)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=args.bs, sampler=train_sampler, drop_last=True)
    test_loader = DataLoader(dataset, batch_size=test_bs, sampler=test_sampler, drop_last=True)

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

    adam_eps = 3e-5 if args.dtype == "float16" else 1e-8
    optimizer = optim.AdamW(model_para, lr=args.lr, eps=adam_eps)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    loss_fn = nn.MSELoss()
    cos_loss = DirectionLoss('cosine')

    if "sld" in args.defense:
        _, _, _, _, _, sld_pipe = get_sd_model(args, "1-5-safe")
    else:
        sld_pipe = None

    ga_bs = 1
    eval_losses = []

    if args.attack == "vae":
        sd_vae = pipe.vae.to(device)
        sd_vae.train()
        for param in sd_vae.parameters():
            param.requires_grad = True

    print("Start training.")
    start_time = time.time()
    with torch.enable_grad():
        torch.cuda.empty_cache()
        for step in range(args.epoch):
            train_loss_ep = 0
            optimizer.zero_grad()
            epoch_start_time = time.time()
            for iter, (images, prompts, masks, targets, sample_i) in enumerate(train_loader):
                with torch.no_grad():
                    prompts = list(prompts)
                    negative_prompt = None
                    if "3-0" in args.version:
                        prompt_embedding, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                            pipe.encode_prompt(prompts, prompt_2=None, prompt_3=None, device=device,
                                               num_images_per_prompt=num_images_per_prompt,
                                               do_classifier_free_guidance=True)
                        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                        prompt_embedding = (prompt_embedding, pooled_prompt_embeds)
                    else:
                        prompt_embedding, negative_prompt_embeds = pipe.encode_prompt(prompts, device=device,
                                                                                  num_images_per_prompt=num_images_per_prompt,
                                                                                  negative_prompt=negative_prompt,
                                                                                  do_classifier_free_guidance=True)
                if args.noise_gen == "adv_noise":
                    noisy_imgs = noise_generator + images
                elif args.noise_gen != "vae":
                    noisy_imgs = noise_generator(images)
                else:
                    noisy_imgs = noise_generator(images).sample

                if args.defense == "noise" or "nadp" in args.attack:
                    noisy_imgs = noisy_imgs + torch.clamp(torch.randn_like(noisy_imgs[0])* (0.05 ** 0.5), max=1, min=-1)

                if args.noi_type == "constrain":
                    constrained_noisy_imgs = torch.min(torch.max(noisy_imgs, images - args.eps),
                                                  images + args.eps)
                else:
                    constrained_noisy_imgs = noisy_imgs

                if args.attack == "ddpm":
                    z_p, targets = DDPM_process(pipe, scheduler, args, constrained_noisy_imgs, prompt_embedding, negative_prompt_embeds,
                                               torch_device, diff_noise_gen, args.bs, num_images_per_prompt, defense=args.defense, get_target=False, latents=targets)
                elif args.attack == "vae":
                    z_p = sd_vae.encode(constrained_noisy_imgs).latent_dist.sample()
                    z_p = sd_vae.decode(z_p, return_dict=False, generator=diff_noise_gen)[0]
                else:
                    z_p, noise_pred, noise_pred_list = diff_process(pipe, scheduler, args, constrained_noisy_imgs, prompt_embedding, negative_prompt_embeds,
                                                   torch_device, diff_noise_gen, args.bs, num_images_per_prompt, defense=args.defense, get_target=False, mask_image=masks)

                loss = loss_fn(z_p, targets) / ga_bs

                del prompt_embedding, noisy_imgs, constrained_noisy_imgs, noise_pred, noise_pred_list
                torch.cuda.empty_cache()

                if args.defense == "sc" or "sc" in args.attack:
                    diff_img = pipe.vae.decode(z_p / pipe.vae.config.scaling_factor, return_dict=False,
                                               generator=diff_noise_gen)[0]
                    sc_loss = sc_loss_factor * get_safety_checker_loss(pipe, diff_img, device, dtype, surrogate_sc)
                    loss += sc_loss

                loss.backward()

                if (iter + 1) % ga_bs == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss_ep += loss.item() * ga_bs

            epoch_end_time = time.time()

            one_ep_time = epoch_end_time - epoch_start_time
            time_s = timedelta(seconds=one_ep_time)
            print(iter, time_s)

            train_loss_ep /= (iter + 1)
            lr_scheduler.step()

            if step % 2 == 0:
                print(f"Epoch {step}, Loss: {train_loss_ep}")
                wandb.log({"Epoch": step, "Training loss":train_loss_ep})

            if step % 20 == 0:
                print("Start testing.")
                if args.noise_gen != "adv_noise":
                    noise_generator.eval()
                with torch.no_grad():
                    eval_loss = 0
                    for iter, (images, prompts, masks, targets, sample_i) in enumerate(test_loader):
                        with precision_scope():
                            # test noise
                            prompts = list(prompts)
                            if args.defense == "ng":
                                if args.concept == "nudity":
                                    negative_prompt = ["naked, nude, bare, exposed, Stripped, topless"]
                                elif args.concept == "violence":
                                    negative_prompt = [
                                        "hate, harassment, violence, suffering, humiliation, harm, suicide, blood"]
                                else:
                                    negative_prompt = [""]
                                negative_prompt = negative_prompt * test_bs
                            else:
                                negative_prompt = None
                            if "3-0" in args.version:
                                prompt_embedding, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                                    pipe.encode_prompt(prompts, prompt_2=None, prompt_3=None, device=device,
                                                       num_images_per_prompt=num_images_per_prompt,
                                                       do_classifier_free_guidance=True)
                                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                                prompt_embedding = (prompt_embedding, pooled_prompt_embeds)
                            else:
                                prompt_embedding, negative_prompt_embeds = pipe.encode_prompt(prompts, device=device,
                                                                                              num_images_per_prompt=num_images_per_prompt,
                                                                                              negative_prompt=negative_prompt,
                                                                                              do_classifier_free_guidance=True)

                            if args.noise_gen == "adv_noise":
                                noisy_imgs = noise_generator + images
                            elif args.noise_gen != "vae":
                                noisy_imgs = noise_generator(images)
                            else:
                                noisy_imgs = noise_generator(images).sample

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
                                                           test_bs, num_images_per_prompt, defense=args.defense, get_target=False, sld_pipe=sld_pipe,
                                                                            mask_image=masks)

                            eval_loss += loss_fn(z_p, targets) / ga_bs

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

                        if args.noise_gen != "adv_noise":
                            torch.save(noise_generator.state_dict(),
                                       f"{output_path}noise/{args.target_pos}_{args.tar_steps}_{args.noi_type}_eps_{scaled_eps}_lr{args.lr}.pt")
                        elif args.noise_gen == "adv_noise":
                            torch.save(noise_generator,
                                       f"{output_path}noise/{args.target_pos}_{args.tar_steps}_{args.noi_type}_eps_{scaled_eps}_lr{args.lr}.pt")

                        if step % 50 == 0:
                            constrained_noisy_imgs = pipe.image_processor.postprocess(constrained_noisy_imgs.detach().cpu(),
                                                                                     output_type="pil", do_denormalize=do_denormalize)
                            for idx, i in enumerate(constrained_noisy_imgs):
                                sample_idx = sample_i[idx]
                                i.save(f"{input_img_save_path}{args.noise_gen}_input_img{sample_idx}.jpg")


                    eval_loss = eval_loss / (iter+1) * ga_bs / test_bs
                    eval_losses.append(eval_loss)

                    if args.concept == "nudity":
                        ASR, failed_sample = nude_test(data_path=f"{img_save_path}")
                    elif args.concept == "violence":
                        ASR, failed_sample = unsafe_test(data_path=f"{img_save_path}", device=device)


                    ## Test on unseen images
                    test_loss = 0
                    for img_idx, images in enumerate(test_images):
                        masks = test_masks[img_idx].to(device)
                        target_idx = len(train_imgs) + img_idx
                        targets = all_targets[target_idx][prompt_idx].unsqueeze(0).to(device)

                        images = images.unsqueeze(0).to(device)
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
                                negative_prompt = negative_prompt * test_bs
                            if "3-0" in args.version:
                                prompt_embedding, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                                    pipe.encode_prompt(prompts, prompt_2=None, prompt_3=None, device=device,
                                                       num_images_per_prompt=num_images_per_prompt,
                                                       do_classifier_free_guidance=True)
                                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                                prompt_embedding = (prompt_embedding, pooled_prompt_embeds)
                            else:
                                prompt_embedding, negative_prompt_embeds = pipe.encode_prompt(prompts, device=device,
                                                                                              num_images_per_prompt=num_images_per_prompt,
                                                                                              negative_prompt=negative_prompt,
                                                                                              do_classifier_free_guidance=True)

                            if args.noise_gen == "adv_noise":
                                noisy_imgs = noise_generator + images
                            elif args.noise_gen != "vae":
                                noisy_imgs = noise_generator(images)
                            else:
                                noisy_imgs = noise_generator(images).sample

                            if args.defense == "noise":
                                noisy_imgs = noisy_imgs + torch.clamp(torch.randn_like(noisy_imgs[0]), max=args.eps,
                                                                      min=-args.eps)

                            if args.noi_type == "constrain":
                                constrained_noisy_imgs = torch.min(torch.max(noisy_imgs, images - args.eps),
                                                                   images + args.eps)
                            else:
                                constrained_noisy_imgs = noisy_imgs


                            z_p, noise_pred, noise_pred_list = diff_process(pipe, copy.deepcopy(scheduler), args, constrained_noisy_imgs,
                                                           prompt_embedding, negative_prompt_embeds, torch_device,
                                                           diff_noise_gen, test_bs, num_images_per_prompt,
                                                           defense=args.defense, get_target=False, sld_pipe=sld_pipe, mask_image=masks, )

                            test_loss += loss_fn(z_p, targets) / ga_bs


                        diff_img = pipe.vae.decode(z_p / pipe.vae.config.scaling_factor, return_dict=False,
                                                   generator=diff_noise_gen)[0]

                        if args.defense == "sc":
                            diff_img, has_nsfw_concept = pipe.run_safety_checker(diff_img, device, dtype)

                        do_denormalize = [True] * diff_img.shape[0]
                        imgs = pipe.image_processor.postprocess(diff_img.detach().cpu(), output_type="pil",
                                                               do_denormalize=do_denormalize)
                        for idx, i in enumerate(imgs):
                            i.save(f"{unseen_img_save_path}{args.noise_gen}_test_img{img_idx}.jpg")

                    test_loss = test_loss / len(test_images)

                    if args.concept == "nudity":
                        test_ASR, Test_failed_sample = nude_test(data_path=f"{unseen_img_save_path}")
                    elif args.concept == "violence":
                        test_ASR, Test_failed_sample = unsafe_test(data_path=f"{unseen_img_save_path}", device=device)

                    txt_out = f'{output_path}{args.noise_gen}_{args.target_pos}_{args.tar_steps}_{args.noi_type}_eps_{scaled_eps}_lr{args.lr}.out'
                    print(f"Step: {step}, Eval loss: {eval_loss}, Test loss: {test_loss}, ASR: {ASR}, "
                          f"Failed sample: {failed_sample}, \n Test ASR: {test_ASR}, Test Failed sample: {Test_failed_sample}")

                    wandb.log({"Epoch": step, "Eval Loss": eval_loss, "ASR": ASR, "Test Loss": test_loss, "Test ASR": test_ASR})

                    with open(txt_out, 'a') as file:
                        file.write(f"Epoch: {step}\n")
                        file.write(f"Args: {args}\n")
                        file.write(f"Eval loss: {eval_loss}\n")
                        file.write(f"Eval ASR: {ASR}\n")
                        file.write(f"Test loss: {test_loss}\n")
                        file.write(f"Test ASR: {test_ASR}\n")
                        file.write(f"Failed sample: {failed_sample}\n")
                        file.write(f"Test Failed sample: {Test_failed_sample}\n")
                    if args.noise_gen != "adv_noise":
                        noise_generator.train()
                del prompt_embedding, noisy_imgs, constrained_noisy_imgs, z_p, diff_img, imgs, targets
                torch.cuda.empty_cache()

    end_time = time.time()
    wandb.finish()

    elapsed_time = end_time - start_time
    elapsed_time_minutes = timedelta(seconds=elapsed_time)
    print(f"Running time:{elapsed_time_minutes}")
    print(f"Your samples are ready and waiting for you here: \n{output_path} \nEnjoy.")


if __name__ == "__main__":
    main()
