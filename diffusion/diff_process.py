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
        sld_warmup_steps = 3
        sld_threshold = 0.025
        sld_momentum_scale = 0.5
        sld_mom_beta = 0.7
    elif safe_strength == "Medium":
        sld_guidance_scale = 1000
        sld_warmup_steps = 5
        sld_threshold = 0.01
        sld_momentum_scale = 0.3
        sld_mom_beta = 0.4
    elif safe_strength == "p2p" or safe_strength == "inpaint":
        sld_guidance_scale = 1000
        sld_warmup_steps = 7
        sld_threshold = 0.01
        sld_momentum_scale = 0.3
        sld_mom_beta = 0.4

    return sld_threshold, sld_warmup_steps, sld_momentum_scale, sld_guidance_scale, sld_mom_beta

def diff_process(pipe, scheduler, args, imgs, prompt_embedding, negative_prompt_embeds, device, generator,
                 batch_size=1, num_images_per_prompt=1, defense="none", **kwargs):

    pipe._interrupt = False
    pipe._guidance_scale = args.scale
    pipe._clip_skip = None
    pipe._cross_attention_kwargs = None
    emb_num = 2
    generator.reset()
    get_target = kwargs["get_target"]
    noise_pred_list = []

    imgs = imgs.to(dtype=pipe.text_encoder.dtype)
    if "3-0" in args.version:
        (prompt_embedding, pooled_prompt_embeds) = prompt_embedding

    prompt_embedding = prompt_embedding.to(dtype=pipe.text_encoder.dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=pipe.text_encoder.dtype)

    if "sld_pipe" not in kwargs.keys():
        sld_pipe = None
    else:
        sld_pipe = kwargs["sld_pipe"]

    if sld_pipe is not None:
        sld_pipe.to(device)
        strength = defense.split("sld_", 1)[1]
        sld_threshold, sld_warmup_steps, sld_momentum_scale, sld_guidance_scale, sld_mom_beta = get_SLD_para(strength)
        safety_embeddings = sld_pipe._encode_prompt(args.concept, device,1,True,"",True).chunk(3)[2]
    if args.version == "p2p":
        pipe._image_guidance_scale = 1
        latents = None

        image_latents = pipe.prepare_image_latents(
            imgs,
            batch_size,
            num_images_per_prompt,
            prompt_embedding.dtype,
            device,
            pipe.do_classifier_free_guidance,
        )
        if sld_pipe is not None:
            emb_num = 4
            # print(prompt_embedding.shape, safety_embeddings.shape)
            prompt_embedding = torch.cat((prompt_embedding, safety_embeddings.repeat(batch_size,1,1)))
            image_latents_single = image_latents.chunk(3)[0]
            image_latents = torch.cat((image_latents, image_latents_single))
            safety_momentum = None
        else:
            emb_num = 3

        height, width = image_latents.shape[-2:]
        height = height * pipe.vae_scale_factor
        width = width * pipe.vae_scale_factor

        # if inference or get_target:
        # 5. set timesteps
        num_steps = args.ddim_steps if get_target else args.tar_steps
        timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_steps, device)
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = pipe.vae.config.latent_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embedding.dtype,
            device,
            generator,
            latents,
        )
        latents.detach()
        noise_pred = latents

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, args.ddim_eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (None)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
        pipe._num_timesteps = len(timesteps)

        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * emb_num) if pipe.do_classifier_free_guidance else latents
                scaled_latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

                # predict the noise residual
                # print(scaled_latent_model_input.shape, prompt_embedding.shape)

                # with torch.cuda.amp.autocast(enabled=False):
                # with torch.no_grad():
                t = t.to(dtype=latents.dtype)

                noise_pred = pipe.unet(
                        scaled_latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embedding,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if pipe.do_classifier_free_guidance:
                    if sld_pipe is not None:

                        noise_pred_text, noise_pred_image, noise_pred_uncond, noise_pred_safety_concept = noise_pred.chunk(emb_num)
                        noise_guidance = noise_pred_text - noise_pred_image
                        if safety_momentum is None:
                            safety_momentum = torch.zeros_like(noise_guidance)

                        # Equation 6
                        scale = torch.clamp(
                            torch.abs((noise_pred_text - noise_pred_safety_concept)) * sld_guidance_scale, max=1.0
                        )

                        # scale = torch.clamp(
                        #     torch.abs((noise_guidance - noise_pred_safety_concept)) * sld_guidance_scale, max=1.0
                        # )

                        # Equation 6
                        safety_concept_scale = torch.where(
                            (noise_pred_text - noise_pred_safety_concept) >= sld_threshold,
                            torch.zeros_like(scale),
                            scale,
                        )

                        # Equation 4
                        noise_guidance_safety = torch.mul(
                            (noise_pred_safety_concept - noise_pred_uncond), safety_concept_scale
                        )

                        # Equation 7
                        noise_guidance_safety = noise_guidance_safety + sld_momentum_scale * safety_momentum

                        # Equation 8
                        safety_momentum = sld_mom_beta * safety_momentum + (1 - sld_mom_beta) * noise_guidance_safety

                        if i >= sld_warmup_steps:  # Warmup
                            # Equation 3
                            noise_guidance = noise_guidance - noise_guidance_safety

                        noise_pred = (
                                noise_pred_uncond
                                + pipe.guidance_scale * noise_guidance
                                + pipe.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        )
                    else:
                        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(emb_num)
                        noise_pred = (
                                noise_pred_uncond
                                + pipe.guidance_scale * (noise_pred_text - noise_pred_image)
                                + pipe.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        )
                    noise_pred_uncond.detach()
                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()

                if args.loss_type == "alleps":
                    noise_pred_list.append(noise_pred)
                # noise_pred_list.append([i for i in noise_pred])

        if args.loss_type == "alleps":
            noise_pred_list = torch.stack(noise_pred_list, dim=1)
        else:
            noise_pred_list = None
        if args.loss_type != "outeps" and get_target == False:
            noise_pred = None

    elif "3-0" in args.version:
        mask_image = kwargs["mask_image"]

        if pipe.do_classifier_free_guidance:
            prompt_embedding = torch.cat([negative_prompt_embeds, prompt_embedding])

        # 4. set timesteps
        num_steps = args.ddim_steps
        strength = args.strength
        width = 512
        height = 512

        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_steps, device)
        timesteps, num_inference_steps = pipe.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )

        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 6. Prepare latent variables
        num_channels_latents = pipe.vae.config.latent_channels
        num_channels_transformer = pipe.transformer.config.in_channels
        return_image_latents = num_channels_transformer == 16

        latents_outputs = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embedding.dtype,
            device,
            generator,
            None,
            image=imgs,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask_condition = pipe.mask_processor.preprocess(
            mask_image.to(torch.float32), height=height, width=width
        )

        masked_image = imgs * (mask_condition < 0.5)
        masked_image = masked_image.to(pipe.dtype)
        mask, masked_image_latents = pipe.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size,
            num_images_per_prompt,
            height,
            width,
            prompt_embedding.dtype,
            device,
            generator,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        )

        # 10. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
        pipe._num_timesteps = len(timesteps)
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                timestep = t.expand(latent_model_input.shape[0])

                if num_channels_transformer == 33:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embedding,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                # perform guidance
                if pipe.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if num_channels_transformer == 16:
                    init_latents_proper = image_latents
                    if pipe.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = pipe.scheduler.scale_noise(
                            init_latents_proper, torch.tensor([noise_timestep]), noise
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()

    elif "inpaint" in args.version:
        mask_image = kwargs["mask_image"]

        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, args.ddim_eta)

        if pipe.do_classifier_free_guidance:
            prompt_embedding = torch.cat([negative_prompt_embeds, prompt_embedding])

        # 4. set timesteps
        num_steps = args.ddim_steps
        strength = args.strength
        width = 512
        height = 512

        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_steps, device)
        timesteps, num_inference_steps = pipe.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )

        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 6. Prepare latent variables
        num_channels_latents = pipe.vae.config.latent_channels
        num_channels_unet = pipe.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        latents_outputs = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embedding.dtype,
            device,
            generator,
            None,
            image=imgs,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask_condition = pipe.mask_processor.preprocess(
            mask_image.to(torch.float32), height=height, width=width
        )

        masked_image = imgs * (mask_condition < 0.5)
        masked_image = masked_image.to(pipe.dtype)
        mask, masked_image_latents = pipe.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embedding.dtype,
            device,
            generator,
            pipe.do_classifier_free_guidance,
        )

        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
        pipe._num_timesteps = len(timesteps)
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embedding,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=pipe.cross_attention_kwargs,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]

                # perform guidance
                if pipe.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if num_channels_unet == 4:
                    init_latents_proper = image_latents
                    if pipe.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = pipe.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()

    else:
        pipe.scheduler = scheduler
        pipe.strength = args.strength

        # if inference or get_target:
        # 5. set timesteps
        # num_steps = args.ddim_steps if get_target else args.tar_steps
        num_steps = args.ddim_steps
        # images = pipe(prompt=["hasdasd"], image=imgs[0], strength=args.strength, guidance_scale=7.5, num_inference_steps=num_steps).images
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

        prompt_embeds.detach()
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

                t = t.to(dtype=latents.dtype)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * emb_num) if pipe.do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                # with torch.no_grad():

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

    return latents, noise_pred, noise_pred_list