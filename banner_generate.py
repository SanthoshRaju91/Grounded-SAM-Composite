import os
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from google.auth import credentials
from PIL import Image, ImageDraw, ImageFont
import io

service_account_key_path = "/home/santhosh.nagaraj/ai-box/Grounded-SAM-Composite/vertex-ai-service-account.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path

vertexai.init(project="mliops-prod", location="us-central1")
model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

local_temp_dir = "/tmp/generations"

def generate_ad_banners(job_id, composite_image, logo, title, catchphrase, template, prompt):
    print(f"::: Generating backgrounds from image gen")
    images = model.generate_images(
        prompt=prompt,
        number_of_images=2,
        language="en",    
        aspect_ratio="16:9",
        safety_filter_level="no nudity, no abuse, no racism",
    )
    print(f"::: Finished generating backgrounds")

    generation_dir = f"{local_temp_dir}/{job_id}"
    generated_bg_images_path = []

    print(f"::: Saving the background images")
    
    for idx, image in enumerate(images):
        local_image_path = f"{generation_dir}/generated_background_{idx}.png"
        image.save(location=local_image_path, include_generation_parameters=False)
        generated_bg_images_path.append(local_image_path)

    print(f"::: Background images saved")
    print(":"*100)

    print(f"::: Creating ad banners")
    for idx, image in enumerate(generated_bg_images_path):
        create_banner(
            image,
            title,
            catchphrase,
            composite_image,
            template,
            job_id,
            idx
        )
    print(f"::: Finished creating ad banners")


def create_banner(local_image_path, title, catchphrase, segmented_image, template, job_id, gen_id):
    print(f"::: Create ad banner for {local_image_path}")
    image = Image.open(local_image_path)
    image_width, image_height = image.size
    
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    font_path = "NotoSansJP-Bold.ttf"
    
    title_font = ImageFont.truetype(font_path, 80)
    
    # Calculate the bounding box of the text
    bbox = draw.textbbox((0,0), title, font=title_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (image_width - text_width) // 2
    y = 140

    # Define padding around the text
    padding = 10
    bottom_padding = 30

    # Calculate the position and size of the border box
    box_x0 = x - padding
    box_y0 = y - padding
    box_x1 = x + text_width + padding
    box_y1 = y + text_height + padding + bottom_padding

    # Draw the border box (rectangle)
    border_color = "white"
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], outline=border_color, width=5)

    # Offsets for the border (to create a "thick" border, you draw the text multiple times)
    offsets = [(-2, -2), (-2, 2), (2, -2), (2, 2), (-2, 0), (2, 0), (0, -2), (0, 2)]
    
    for offset in offsets:
        draw.text((x + offset[0], y + offset[1]), title, font=title_font, fill=border_color)

    # Draw the main text in a different color (e.g., black)
    text_color = "white"
    draw.text((x, y), title, font=title_font, fill=text_color)

    ### Catchphrase
    catchphrase_front = ImageFont.truetype(font_path, 48)

    # Calculate the size of the text
    bbox = draw.textbbox((0, 0), catchphrase, font=catchphrase_front)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (image_width - text_width) // 2
    y = 240


    for offset in offsets:
        draw.text((x + offset[0], y + offset[1]), catchphrase, font=catchphrase_front, fill=border_color)

    # Draw the main text in a different color (e.g., black)
    text_color = "white"
    draw.text((x, y), catchphrase, font=catchphrase_front, fill=text_color)

    segmented_image = Image.open(segmented_image)
    product_image_meta = template["product_image"]
    
    width = product_image_meta["size"][0]
    height = product_image_meta["size"][1]

    x = product_image_meta["position"][0]
    y = product_image_meta["position"][1]
    
    segmented_image = segmented_image.resize((width, height))

    banner = image.resize((800, 320))
    banner.paste(segmented_image, (x, y), segmented_image)
    banner.save(f"{local_temp_dir}/{job_id}/ad_banner_{gen_id}.png")
    

    


