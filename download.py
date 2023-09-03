# This file runs during container build time to get model weights built into the container
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model_id = "timbrooks/instruct-pix2pix"
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)

if __name__ == "__main__":
    download_model()