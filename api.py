import os
import io
import base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
import json

from segment import GroundedSAMComposite
from prompt_fine_tuner import generate_product_prompt
from download_image import download_image
from banner_generate import generate_ad_banners
from update_status import update_status

app = FastAPI()

model = GroundedSAMComposite()

items = None
styles = None
templates = None
local_temp_dir = "/tmp/generations"

def read_json_file(file_path):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def count_trues(tup):
    return sum(1 for item in tup if item is True)


def create_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        print(f"An error occurred while creating the directory: {str(e)}")
        

@app.on_event("startup")
async def load_json_files():
    global items
    global styles
    global templates

    items = read_json_file("resources/items_db.json")
    styles = read_json_file("resources/styles.json")
    templates = read_json_file("resources/templates.json")


class GeneratePayload(BaseModel):
    id: int
    title: str
    catchphrase: str
    banner_id: str
    style: str
    logo: str
    item_url: str

@app.get("/server/health")
def server_health():
    return "ok"


@app.get("/server/version")
def server_version():
    version = os.getenv("VERSION", "devlocal-0.0.0")
    return {
        "version": version
    }
    

@app.post("/api/generate")
async def predict(payload: GeneratePayload):
    job_id = payload.id
    banner_id = payload.banner_id
    style = payload.style
    logo = payload.logo
    item_url = payload.item_url
    title = payload.title
    catchphrase = payload.catchphrase

    print(f"::: Got the request for job_id: {job_id}")

    item = items[item_url]
    background_prompt = styles[style]
    banner_template = templates[banner_id]

    print(f"::: Getting the segmentation prompt")    
    segmentation_prompt = generate_product_prompt(item["item_description"], item["item_name"])
    print(f"::: Got the segmentation prompt: {segmentation_prompt}")
    
    images = item["images"]
    local_images = []
    generation_dir = f"{local_temp_dir}/{job_id}"

    print()
    create_dir(generation_dir)

    print(f"::: Downloading the images for the item: {item_url}")
    for idx, image in enumerate(images):        
        local_image_path = f"{generation_dir}/image_{idx}.jpg"
        download_image(image, local_image_path)
        local_images.append(local_image_path)
    print(f"::: Finished downloading images")

    print(f"::: Performing segmentation and BPIP checks")
    predicts = []    
    for local_image in local_images:
        is_centered, is_background_uniform, is_sufficient_contrast, composite_image = model.predict(local_image_path=local_image, prompt=segmentation_prompt)
        predicts.append((
            local_image,
            is_centered,
            is_background_uniform,
            is_sufficient_contrast,
            composite_image
        ))
    print(f"::: Finished segmentations and BPIP checks")

    print(f"::: Performing ranking")
    appropriate = max(predicts, key=count_trues)

    print(f"::: Finished BPIP ranking, saving the best product segmented image")
    composite_image = appropriate[4]
    composite_image_path = f"{generation_dir}/segmented_image.png"
    composite_image.save(composite_image_path, format="PNG")
    print(f"::: Saved the product's segmented image")

    print(f"::: Generating ad banners")
    saved_images = generate_ad_banners(job_id, composite_image_path, logo, title, catchphrase, banner_template, background_prompt)
    update_status(job_id, saved_images)

    return "ok"



def pil_image_to_base64(image: Image.Image) -> str:
    # Create an in-memory binary stream
    buffered = io.BytesIO()
    
    # Save the image in the buffer as PNG (or any other format)
    image.save(buffered, format="PNG")
    
    # Encode the image to base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="api:app", host="0.0.0.0", port=8000)