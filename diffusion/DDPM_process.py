import torch
import inspect

from utils.util import retrieve_timesteps

# @torch.no_grad()
# @replace_example_docstring(EXAMPLE_DOC_STRING)
def get_SLD_para(safe_strength="MAX"):
    if safe_strength == "Max":
        sld_guidance_scale = 5000
        sld_warmup_steps = 0
        sld_threshold = 1.0
        sld_momentum_scale = 0.5
        sld_mom_beta = 0.7
    elif safe_strength == "Strong":
        sld_guidance_scale = 2000
        sld_warmup_steps = 7
        sld_threshold = 0.025
        sld_momentum_scale = 0.5
        sld_mom_beta = 0.7
    elif safe_strength == "Medium":
        sld_guidance_scale = 1000
        sld_warmup_steps = 10
        sld_threshold = 0.01
        sld_momentum_scale = 0.3
        sld_mom_beta = 0.4

    return sld_threshold, sld_warmup_steps, sld_momentum_scale, sld_guidance_scale, sld_mom_beta


def DDPM_process(pipe, scheduler, args, imgs, prompt_embedding, negative_prompt_embeds, device, generator,
                 batch_size=1, num_images_per_prompt=1, defense="none", **kwargs):

    pipe._interrupt = False
    pipe._guidance_scale = args.scale
    pipe._clip_skip = None
    pipe._cross_attention_kwargs = None

    emb_num = 2
    generator.reset()

    get_target = kwargs["get_target"]
    latents = kwargs["latents"]

    if args.version != "p2p":
        pipe.strength = args.strength

        # if inference or get_target:
        # 5. set timesteps
        num_steps = args.ddim_steps if get_target else args.tar_steps
        timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_steps, device)
        timesteps, num_inference_steps = pipe.get_timesteps(num_inference_steps, pipe.strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        pipe._num_timesteps = len(timesteps)
        # 6. Prepare latent variables
        latents = pipe.prepare_latents(imgs, latent_timestep,
                                       batch_size, num_images_per_prompt, prompt_embedding.dtype,
                                       device, generator=generator)
        noise_pred = latents
        if pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embedding])
        else:
            prompt_embeds = prompt_embedding
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, args.ddim_eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (None)

        # 7.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order

        noise_pred_list = []
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * emb_num) if pipe.do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=pipe.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if pipe.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(emb_num)
                    noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
                noise_pred_list.append(noise_pred)
        noise_pred_list = torch.stack(noise_pred_list, dim=1)

    else:
        pipe._image_guidance_scale = 1
        latents = latents * pipe.vae.config.scaling_factor
        prompt_embedding = prompt_embedding.chunk(3)[0]

        image_latents = pipe.prepare_image_latents(
            imgs,
            batch_size,
            num_images_per_prompt,
            prompt_embedding.dtype,
            device,
            do_classifier_free_guidance=False,
        )

        emb_num = 1

        added_cond_kwargs = (None)

        bsz = latents.shape[0]
        timesteps_idx = torch.randint(0, args.tar_steps, (bsz,), device=latents.device)
        # timesteps = timesteps.long()

        num_steps = args.ddim_steps if get_target else args.tar_steps
        timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_steps, device)
        timesteps = timesteps[timesteps_idx]
        # pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = pipe.scheduler.timesteps

        # idx = torch.randint(0, len(timesteps), (args.bs,))
        # timesteps = timesteps[idx]

        # noise = torch.randn_like(latents)
        noise = torch.randn(latents.shape, generator=generator, dtype=latents.dtype).to(device)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        target = noise
        # expand the latents if we are doing classifier free guidance
        # scaled_latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timesteps)
        # scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

        scaled_latent_model_input = torch.cat([noisy_latents, image_latents], dim=1)

        timesteps = timesteps.repeat(emb_num)
        # predict the noise residual
        # print(scaled_latent_model_input.shape, prompt_embedding.shape)
        noise_pred = pipe.unet(
            scaled_latent_model_input,
            timesteps,
            encoder_hidden_states=prompt_embedding,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]


    return noise_pred, target