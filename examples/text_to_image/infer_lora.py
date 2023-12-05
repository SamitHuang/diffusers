import os
from diffusers import AutoPipelineForText2Image
import torch
from fire import Fire


def infer(prompt="a very cute looking pokemon with big eyes", 
        model_path="sd-cnart-model-lora-sdxl", seed=42, images_per_prompt = 4):
    
    # model_path = "sd-pokemon-model-lora-sdxl"
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
	"models/sdxl_base", torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    pipeline_text2image.unet.load_attn_procs(model_path)

    generator = [torch.Generator(device="cuda").manual_seed(seed+i) for i in range(images_per_prompt)]

    output_dir = model_path + "-gen-images" # "lora_ft_outputs" 
    os.makedirs(output_dir, exist_ok=True)

    # prompt = "a painting of a tree with a mountain in the background and a person standing in the foreground with a snow covered ground"
    #image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    # for i in range(n_cols):
    images = pipeline_text2image(prompt=prompt, generator=generator, num_inference_steps=40, height=1024, width=1024, num_images_per_prompt=images_per_prompt).images
    
    for i in range(images_per_prompt): 
        images[i].save("{}/cnart_test_3_idx{}.png".format(output_dir, i))
    
    print("Generated images are saved in ", output_dir)

if __name__ == "__main__":
    Fire(infer)
