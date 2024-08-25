import os
import io
import base64
from fastapi import FastAPI, File, UploadFile
import tempfile
import shutil
from PIL import Image

from segment import GroundedSAMComposite

app = FastAPI()

model = GroundedSAMComposite()

@app.get("/server/health")
def server_health():
    return "ok"


@app.get("/server/version")
def server_version():
    version = os.getenv("VERSION", "devlocal-0.0.0")
    return {
        "version": version
    }
    

@app.post("/api/predict")
async def predict(prompt: str, file: UploadFile = File(...)):
     # Create a temporary file with NamedTemporaryFile
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Copy the content of the uploaded file to the temporary file
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name  # Store the path to the temporary file

    composite_image = model.predict(tmp_path, prompt)
    
    image_str = pil_image_to_base64(composite_image)
    return {
        "segmented_image": image_str
    }



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
    uvicorn.run(app=app, host="0.0.0.0", port=8000)