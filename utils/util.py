import math
import numpy as np
import torch
import pandas as pd
import os
import torch.nn as nn
import inspect
import argparse
from PIL import ImageOps
from PIL import Image
from diffusers import (AutoencoderKL, UNet2DConditionModel, DDPMScheduler, EulerDiscreteScheduler,
                       StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline,
                       StableDiffusionPipelineSafe, StableDiffusionPipeline, StableDiffusionInpaintPipeline,
                       StableDiffusion3InpaintPipeline, StableDiffusion3Pipeline, AutoPipelineForImage2Image,
                       FluxPipeline)
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange, repeat
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from typing import Any, Callable, Dict, List, Optional, Union
from utils.text_docs import *
from utils.test_detector import nude_test, unsafe_test


class DiffuisonNoiseGenerator(torch.Generator):
    def __init__(self):
        super().__init__()
        self.fixed_state = None
        self._initialize_state()

    def _initialize_state(self):
        self.fixed_state = self.get_state()

    def reset(self):
        self.set_state(self.fixed_state)

class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='cosine'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)

        return self.loss_func(x, y)

def get_sd_model(args, version):
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError
    model_dir = "/data/yaopei/huggingface/hub/"
    if version == '1-4':
        model_id = "CompVis/stable-diffusion-v-1-4-original"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype,
                                                              cache_dir=model_dir)
        vae = AutoencoderKL.from_pretrained(f"CompVis/stable-diffusion-v{args.version}",
                                            subfolder="vae", torch_dtype=dtype, cache_dir=model_dir)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir=model_dir)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype, cache_dir=model_dir)
        unet = UNet2DConditionModel.from_pretrained(f"CompVis/stable-diffusion-v{args.version}",
                                                    subfolder="unet", torch_dtype=dtype, cache_dir=model_dir)
        scheduler_config = get_scheduler_config(args)
        scheduler = DDPMScheduler(num_train_timesteps=scheduler_config['num_train_timesteps'],
                                  beta_start=scheduler_config['beta_start'],
                                  beta_end=scheduler_config['beta_end'],
                                  beta_schedule=scheduler_config['beta_schedule'])
    elif version == '2-1':
        model_id = "stabilityai/stable-diffusion-2-1-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype, cache_dir=model_dir)

        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
    elif version == '2-0':
        model_id = "stabilityai/stable-diffusion-2-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype, cache_dir=model_dir)
        pipe.enable_xformers_memory_efficient_attention()
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
    elif version == '1-5':
        model_id = "runwayml/stable-diffusion-v1-5"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype, cache_dir=model_dir)
        # pipe.enable_xformers_memory_efficient_attention()
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
    elif version == '1-5-safe':
        model_id = "AIML-TUDA/stable-diffusion-safe"
        pipe = StableDiffusionPipelineSafe.from_pretrained(model_id, torch_dtype=dtype, cache_dir=model_dir)
        pipe.safety_concept = ('an image showing hate, harassment, violence, suffering, humiliation, harm, suicide, nudity, bodily fluids, blood, obscene gestures, '
                               'illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty')

        # pipe.safety_concept = ('an image showing nudity')
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe.enable_xformers_memory_efficient_attention()
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
    elif version == '3-0':
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        revision = "main" if args.dtype == "float32" else "fp16"
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,  # put your model path here
            # revision=revision,
            torch_dtype=dtype,
            cache_dir=model_dir
        )
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.transformer
    elif version == 'p2p':
        model_id = "timbrooks/instruct-pix2pix"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=dtype, cache_dir=model_dir)

        # pipe.safety_concept = ('an image showing nudity')
        pipe.enable_xformers_memory_efficient_attention()
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
    elif version == '2-0-inpaint':
        model_id = "stabilityai/stable-diffusion-2-inpainting"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        revision = "main" if args.dtype == "float32" else "fp16"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,  # put your model path here
            revision=revision,
            torch_dtype=dtype,
            cache_dir=model_dir
        )
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
    elif version == '1-5-inpaint':
        # model_id = "benjamin-paine/stable-diffusion-v1-5-inpainting"
        model_id = "botp/stable-diffusion-v1-5-inpainting"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        revision = "main" if args.dtype == "float32" else "fp16"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,  # put your model path here
            # revision=revision,
            torch_dtype=dtype,
            cache_dir=model_dir
        )
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
    elif version == '2-1-inpaint':
        model_id = "HieuPM/stable-diffusion-2-1-inpainting"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        revision = "main" if args.dtype == "float32" else "fp16"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,  # put your model path here
            # revision=revision,
            torch_dtype=dtype,
            cache_dir=model_dir
        )
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet

    elif version == '3-0-inpaint':
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        revision = "main" if args.dtype == "float32" else "fp16"
        pipe = StableDiffusion3InpaintPipeline.from_pretrained(
            model_id,  # put your model path here
            # revision=revision,
            torch_dtype=dtype,
            cache_dir=model_dir
        )
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.transformer
    elif version == 'sd-turbo':
        model_id = "stabilityai/sd-turbo"
        revision = "" if args.dtype == "float32" else "fp16"
        pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=dtype,
                                                         variant=revision)
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
    elif version == 'flux':
        model_id = "black-forest-labs/FLUX.1-dev"
        adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        )
        pipe.to("cuda")
        pipe.load_lora_weights(adapter_id)
        pipe.fuse_lora()
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.transformer

    else:
        raise NotImplementedError

    for param in unet.parameters():
        param.requires_grad = False
    for param in pipe.vae.parameters():
        param.requires_grad = False
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False

    return vae, tokenizer, text_encoder, unet, scheduler, pipe

def get_scheduler_config(args):
    assert args.version in {'1-4', '2-1'}
    if args.version == '1-4':
        config = {
            "_class_name": "PNDMScheduler",
            "_diffusers_version": "0.7.0.dev0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False
        }
    elif args.version == '2-1':
        config = {
            "_class_name": "EulerDiscreteScheduler",
            "_diffusers_version": "0.10.2",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,  # todo
            "trained_betas": None
        }
    else:
        raise NotImplementedError

    return config

def load_and_preprocess_image_mask(path, mask_path=None, resolution=256, pipe=None, device=None, dtype=None):
    input_image = Image.open(path).convert("RGB")
    processed_img = pipe.image_processor.preprocess(input_image).to(dtype)

    input_mask = Image.open(mask_path).convert("L")
    processed_mask = pipe.image_processor.preprocess(input_mask).to(dtype)

    masked_image = processed_img * (processed_mask < 0.5)

    return processed_img, processed_mask, masked_image

def load_and_preprocess_image(path, resolution=256, pipe=None, device=None, dtype=None):
    input_image = Image.open(path).convert("RGB").resize((resolution, resolution))
    processed_img = pipe.image_processor.preprocess(input_image).to(dtype)
    return processed_img

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def get_orig_prompts(prompt_length, concept=None):
    original_prompts = original_prompts_bg

    if prompt_length > len(original_prompts):
        print("length of original prompts: ", len(original_prompts))
        prompts = original_prompts
    else:
        prompts = original_prompts[:prompt_length]

    return prompts


def extract_number_from_filename(filename):
    import re
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def get_orig_prompts_ring(prompt_length, concept):
    df = pd.read_csv('scripts/text_docs/unsafe-prompts4703.csv')
    prompts = []
    if concept in "nudity":
        for _, row in df.iterrows():
            if float(row.nudity_percentage):
                prompts.append(row.prompt)
    select_prompts = prompts[:prompt_length]
    return select_prompts


def get_target(all_images, org_prompts, pipe, args, steer_vec, precision_scope, device, target_path, output_path,
               diff_process, generator, scheduler, num_images_per_prompt=1, masks=None):

    tar_img_save_path = f"{output_path}target/"
    os.makedirs(tar_img_save_path, exist_ok=True)
    batch_size = min(args.bs, len(org_prompts))
    with torch.no_grad(), precision_scope():
        target_save = []
        target_noise_save = []
        target_noise_list_save = []
        target_vae_imag_save =[]
        num_prompts = len(org_prompts)
        for img_idx, img in enumerate(all_images):
            target_per_img = []
            target_noise_per_img = []
            target_noise_list_per_img = []
            vae_diff_img = []

            if masks is not None:
                img_mask = masks[img_idx].unsqueeze(0).repeat(batch_size, 1, 1, 1)
            else:
                img_mask = None
            img_batch = img.unsqueeze(0).repeat(batch_size, 1, 
                                                1, 1)
            print("Process img: ", img_idx)
            for start_idx in range(0, num_prompts, batch_size):
                end_idx = min(start_idx + batch_size, num_prompts)
                num_samples = min(batch_size, num_prompts-start_idx)
                # print(f"Process prompt: {start_idx}~{end_idx}")
                prompt_batch = org_prompts[start_idx:end_idx]
                img_batch = img_batch[:num_samples]
                if masks is not None:
                    img_mask = img_mask[:num_samples]

                if args.version == "p2p":
                    target_embedding = pipe._encode_prompt(prompt_batch, device=device,
                                                           num_images_per_prompt=num_images_per_prompt,
                                                           do_classifier_free_guidance=True)
                    negative_prompt_embeds = None
                    cond_embedding = target_embedding.chunk(3)[0]
                    steered_cond_embedding = cond_embedding + args.vec_scale * steer_vec
                    target_embedding[:num_samples] = steered_cond_embedding
                elif "3-0" in args.version:
                    prompt_batch = ["Make the woman bloody."] * len(prompt_batch)
                    # for i, promt in enumerate(prompt_batch):
                        # prompt_batch[i] = promt + "And bloody."
                    prompt_embedding, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                    pipe.encode_prompt(prompt_batch, prompt_2=prompt_batch, prompt_3=prompt_batch, device=device,
                                                                                  num_images_per_prompt=num_images_per_prompt,
                                                                                  do_classifier_free_guidance=True)

                    # target_embedding = prompt_embedding + args.vec_scale * steer_vec
                    # pooled_prompt_embeds = pooled_prompt_embeds + args.vec_scale * steer_vec
                    target_embedding = prompt_embedding
                    pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                    target_embedding = (target_embedding, pooled_prompt_embeds)
                else:
                    # prompt_batch = ["Make the woman naked."] * len(prompt_batch)
                    prompt_embedding, negative_prompt_embeds = pipe.encode_prompt(prompt_batch, device=device,
                                                                                  num_images_per_prompt=num_images_per_prompt,
                                                                                  do_classifier_free_guidance=True)

                    target_embedding = prompt_embedding + args.vec_scale * steer_vec
                    # target_embedding = prompt_embedding

                latents, noise_pred, tar_noise_pred = diff_process(pipe, scheduler, args, img_batch, target_embedding,
                                                                   negative_prompt_embeds, device, generator,
                                                                   num_samples, num_images_per_prompt, defense=args.defense,
                                                                   get_target=True, mask_image=img_mask)

                target_per_img.append(latents.detach().cpu())
                target_noise_per_img.append(noise_pred.detach().cpu())

                if args.loss_type == "alleps":
                    target_noise_list_per_img.append(tar_noise_pred.detach().cpu())

                diff_img = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[0]

                vae_diff_img.append(diff_img.detach().cpu())

                do_denormalize = [True] * diff_img.shape[0]
                imgs = pipe.image_processor.postprocess(diff_img.detach().cpu(), output_type="pil",
                                                       do_denormalize=do_denormalize)
                for idx, i in enumerate(imgs):
                    i.save(f"{tar_img_save_path}{args.version}_img{img_idx}_p{start_idx + idx}.jpg")
                del latents, noise_pred, tar_noise_pred
                torch.cuda.empty_cache()

            target_per_img = torch.cat(target_per_img, dim=0)
            target_noise_per_img = torch.cat(target_noise_per_img, dim=0)
            vae_diff_per_img = torch.cat(vae_diff_img, dim=0)

            target_save.append(target_per_img)
            target_noise_save.append(target_noise_per_img)

            if args.attack == "vae":
                target_vae_imag_save.append(vae_diff_per_img)

            if args.loss_type == "alleps":
                target_noise_list_per_img = torch.cat(target_noise_list_per_img, dim=0)
                target_noise_list_save.append(target_noise_list_per_img)

    target_save = torch.stack(target_save)
    target_noise_save = torch.stack(target_noise_save)
    if args.attack == "vae":
        target_vae_imag_save = torch.stack(target_vae_imag_save)
        targets = target_vae_imag_save
    else:
        if args.loss_type == "outeps" or args.loss_type == "cos":
            targets = target_noise_save
        elif args.loss_type == "alleps":
            target_noise_list_save = torch.stack(target_noise_list_save)
            targets = target_noise_list_save
        else:
            targets = target_save

    torch.save(targets, target_path)

    print("Get target success.")

    return


def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="",
    )
    parser.add_argument(
        "--eps",
        type=str,
        default="128/255",
        help="noise bound",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
        default=""
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="2-1"
    )
    parser.add_argument(
        "--target_pos",
        type=str,
        default="zT-1"
    )
    parser.add_argument(
        "--tar_steps",
        type=int,
        default=10
    )
    parser.add_argument("--noi_type",type=str,default="constrain")
    parser.add_argument("--vec_scale", type=float, default=1.0, help="scale of steering vector")
    parser.add_argument("--concept",type=str,default="nudity")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--loss_type",type=str,default="outeps")
    parser.add_argument("--prefix",type=str,default="")
    parser.add_argument("--defense",type=str,default="")
    parser.add_argument("--noise_gen",type=str,default="vae")
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--attack", default="", type=str)
    parser.add_argument("--data_type", default="easy", type=str)
    parser.add_argument("--mask_dir", default="naked_imgs_easy_clothes_mask", type=str)

    return parser.parse_args()