from potassium import Potassium, Request, Response

import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import base64
from io import BytesIO
import os

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model_id = "timbrooks/instruct-pix2pix"
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    if torch.cuda.is_available():
        print("cuda is available")
        model.to("cuda")
    model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)
   
    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")
    model_inputs = request.json
    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    image_base64 = model_inputs.get('image', None)
    
    if prompt == None:
        return Response(json={'message': "No prompt provided"},status=400)
    if image_base64 == None:
        return Response(json={'message': "No image provided"},status=400)
    
    steps = model_inputs.get('steps', 30)
    image_guidance = model_inputs.get('image_guidance', 1.5)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    negative_prompt = model_inputs.get('negative_prompt', None)
    image_downloaded = PIL.Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    
    # Run the model
    images = model(prompt, image=image_downloaded, num_inference_steps=steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, image_guidance_scale=image_guidance).images

    # Return the results as a dictionary
    buffered = BytesIO()
    images[0].save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return Response(
        json = {'image_base64': image_base64},
        status=200
    )

if __name__ == "__main__":
    app.serve()