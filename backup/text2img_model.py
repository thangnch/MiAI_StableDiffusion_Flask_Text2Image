import torch
from diffusers import StableDiffusionPipeline

seed = 42
generator = torch.manual_seed(seed)

NUM_INFERENCE_STEPS = 25
NUM_IMAGES_PER_PROMPT = 1
GUIDANCE_SCALE = 0.75
HEIGHT = 512
WIDTH = 512

model_list = ["nota-ai/bk-sdm-small",
              "CompVis/stable-diffusion-v1-4",
              "runwayml/stable-diffusion-v1-5",
              "prompthero/openjourney",
              "hakurei/waifu-diffusion",
              "stabilityai/stable-diffusion-2-1",
              "dreamlike-art/dreamlike-photoreal-2.0"
              ]


def create_pipeline(model_name=model_list[0]):
    if torch.cuda.is_available():
        print("Using GPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("mps")
    else:
        print("Using CPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
    return pipeline


def text2img(prompt, pipeline):
    images = pipeline(
        prompt,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        height=HEIGHT,
        width=WIDTH
    ).images

    return images[0]
