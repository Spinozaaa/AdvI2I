export CUDA_VISIBLE_DEVICES=0

## generate NSFW vector
python -u vec_gen.py --ddim_steps 8 --tar_steps 8 --strength 1 --scale 10 --version 1-5-inpaint --dtype float16 --concept nudity

## train NSFW generator
# python -u opt_generator_inpaint.py --ddim_steps 8 --tar_steps 8 --strength 1 --vec_scale 2.5 --concept nudity --mask_dir naked_imgs_easy_processed \
#  --version 1-5-inpaint --dtype float16 --resolution 512 --bs 1 --epoch 100 --lr 1e-5 --eps 64/255 --loss_type mse --prefix "" 

## eval

# python -u eval_generator_inpaint.py --ddim_steps 8 --tar_steps 8 --strength 1 --vec_scale 2.5 --concept nudity --mask_dir img_clothes_masks \
#  --version 1-5-inpaint --dtype float16 --resolution 512 --bs 1 --epoch 1 --lr 1e-5 --eps 64/255 --loss_type mse --prefix "eval_gen_time" \
#  --ckpt your_checkpoint
 
