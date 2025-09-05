# AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion Models

This repository contains the implementation for the ICML paper "AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion Models".

## Abstract

This work presents AdvI2I, a novel adversarial attack method targeting image-to-image diffusion models. Our approach generates adversarial perturbations that can manipulate the output of diffusion models while maintaining visual quality and semantic coherence.

## Installation

### Environment Setup

1. Create a conda environment:
```bash
conda env create -f environment.yml
conda activate advi2i
```

2. Install additional dependencies:
```bash
pip install -r requirements.txt
```

The nudenet could be installed by:

```
pip install nudenet
```

### Data Preparation

1. Download and extract the `dataset.tar.gz` file, which contains:
   - Training and testing images
   - Mask images for inpainting models
   - Prompts for constructing NSFW concept and dataset

2. Place the extracted data in the following structure:
```
your_path/adv_diffusion/
├── dataset/
│   ├── naked_imgs/          # Training and testing images
│   ├── prompts/             # Prompts for constructing NSFW concept and dataset
│   └── img_clothes_masks/   # Mask images for inpainting
└── hf_results/              # Output directory
```

3. Update the `work_path` in the scripts to point to your data directory.

## Usage

### 1. Generate NSFW concept

First, generate the NSFW concept for your target concept:

```bash
python vec_gen.py --concept nudity --version 1-5-inpaint --dtype float16
```

### 2. Train Adversarial Generators

#### For Inpainting Models:
```bash
python opt_generator_inpaint.py \
    --ddim_steps 8 \
    --tar_steps 8 \
    --strength 1 \
    --vec_scale 2.5 \
    --concept nudity \
    --mask_dir img_clothes_masks \
    --version 1-5-inpaint \
    --dtype float16 \
    --epoch 100 \
    --lr 1e-5 \
    --eps 64/255 \
    --loss_type mse \
    --prefix ""
```

#### For P2P Models:
```bash
python opt_generator_p2p.py \
    --ddim_steps 8 \
    --tar_steps 8 \
    --strength 1 \
    --vec_scale 2.5 \
    --concept nudity \
    --version p2p \
    --dtype float16 \
    --epoch 100 \
    --lr 1e-5 \
    --eps 64/255 \
    --loss_type mse \
    --prefix ""
```

### 3. Evaluate Attack Performance

#### For Inpainting Models:
```bash
python eval_generator_inpaint.py \
    --ddim_steps 8 \
    --tar_steps 8 \
    --strength 1 \
    --vec_scale 2.5 \
    --concept nudity \
    --mask_dir img_clothes_masks \
    --version 1-5-inpaint \
    --dtype float16 \
    --lr 1e-5 \
    --eps 64/255 \
    --loss_type mse \
    --prefix "eval_gen_time" \
    --ckpt your_checkpoint
```

## Parameters

### Key Parameters

- `--concept`: Target concept for attack (nudity, violence)
- `--version`: Model version (1-5-inpaint, 2-1-inpaint, p2p, ...)
- `--vec_scale`: Steering vector scale factor
- `--eps`: Perturbation budget (e.g., 64/255)
- `--lr`: Learning rate for optimization
- `--epoch`: Number of training epochs
- `--bs`: Batch size
- `--dtype`: Data type (float16, float32)

### Attack Methods

- `--noise_gen`: Noise generation method (adv_noise, vae, etc.)
- `--attack`: Attack type (vae in default)
- `--defense`: Defense mechanism (sc, noise, ng, diffpure)

## File Structure

```
├── eval_generator_inpaint.py    # Evaluation script for inpainting models
├── eval_generator_p2p.py        # Evaluation script for P2P models
├── opt_generator_inpaint.py     # Training script for inpainting models
├── opt_generator_p2p.py         # Training script for P2P models
├── vec_gen.py                   # NSFW concept generation
├── run.sh                       # Example run script
├── environment.yml              # Conda environment file
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{zengadvi2i,
  title={AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion Models},
  author={Zeng, Yaopei and Cao, Yuanpu and Cao, Bochuan and Chang, Yurui and Chen, Jinghui and Lin, Lu},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

## Notes

- Make sure to update the `work_path` variable in all scripts to point to your data directory
- The `your_wandb_account` entity in wandb.init() should be replaced with your Weights & Biases account
- For questions or issues, please open an issue on GitHub or contact zengyaopei@gmail.com.
