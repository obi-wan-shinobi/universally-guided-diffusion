# Universally Guided Diffusion

## 1. Environment Set Up

Install dependencies with conda and pip:

```bash
conda env create -f environment.yaml
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install GPUtil
pip install blobfile
pip install facenet-pytorch
```

## 2. Instructions

### Stable Diffusion

Enter the `stable-diffusion` directory.

- **DDPM sampling:**
  ```bash
  python -m scripts.clip_guided \
    --text "Headshot of a person with blonde hair with space background" \
    --ckpt checkpoints/sd-v1-4.ckpt \
    --optim_num_steps 2 \
    --optim_forward_guidance_wt 20000 \
    --optim_original_conditioning
  ```

- **DDIM sampling:**
  ```bash
  python -m scripts.clip_guided_ddim \
    --text "Headshot of a person with blonde hair with space background" \
    --ckpt checkpoints/sd-v1-4.ckpt \
    --optim_num_steps 2 \
    --optim_forward_guidance_wt 20000 \
    --optim_original_conditioning
  ```

### OpenAI Diffusion

Enter the `openai-diffusion` directory.

- **Example usages for the cases in the original paper:**
  ```bash
  python Guided/Clip_guided.py --trials 5 --samples_per_diffusion 2 --text "English foxhound by Edward Hopper" --optim_forward_guidance --optim_forward_guidance_wt 2.0 --optim_num_steps 10 --optim_folder ./Clip_btd_cake/ --batch_size 8 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --model_path <Path to the unconditional diffusion model>

  python Guided/Clip_guided.py --trials 5 --samples_per_diffusion 2 --text "Van Gogh Style" --optim_forward_guidance --optim_forward_guidance_wt 5.0 --optim_num_steps 5 --optim_folder ./Clip_btd_cake/ --batch_size 8 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --model_path <Path to the unconditional diffusion model>

  python Guided/Clip_guided.py --trials 5 --samples_per_diffusion 2 --text "Birthday Cake" --optim_forward_guidance --optim_forward_guidance_wt 2.0 --optim_num_steps 10 --optim_folder ./Clip_btd_cake/ --batch_size 8 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --model_path <Path to the unconditional diffusion model>
  ```
