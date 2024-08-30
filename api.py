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

app = FastAPI()

model = GroundedSAMComposite()

items = None
styles = None
templates = None
local_temp_dir = "/tmp"

def read_json_file(file_path):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def count_trues(tup):
    return sum(1 for item in tup if item is True)


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
    id = payload.id
    banner_id = payload.banner_id
    style = payload.style
    logo = payload.logo
    item_url = payload.item_url

    item = items[item_url]
    background_prompt = styles[style]
    banner_template = templates[banner_id]
    segmentation_prompt = generate_product_prompt(item["item_description"], item["item_name"])
    images = item["images"]
    local_images = []
    for idx, image in enumerate(images):        
        local_image_path = f"{local_temp_dir}/image_{idx}.jpg"
        download_image(image, local_image_path)
        local_images.append(local_image_path)

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
    appropriate = max(predicts, key=count_trues)
    print(appropriate[4])
        
    return "ok"
    #  # Create a temporary file with NamedTemporaryFile
    # with tempfile.NamedTemporaryFile(delete=False) as tmp:
    #     # Copy the content of the uploaded file to the temporary file
    #     shutil.copyfileobj(file.file, tmp)
    #     tmp_path = tmp.name  # Store the path to the temporary file

    # composite_image = model.predict(tmp_path, prompt)
    
    # image_str = pil_image_to_base64(composite_image)
    # return {
    #     "segmented_image": image_str
    # }



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
    uvicorn.run(app="api:app", host="0.0.0.0", port=8000, reload=True)