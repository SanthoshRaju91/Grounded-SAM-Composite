import os
from PIL import Image, ImageDraw, ImageFont
import io

local_temp_dir = "/tmp/generations"

def generate_ad_banners(job_id, composite_image, logo, title, catchphrase, template, prompt):
    print(f"::: Generating backgrounds from image gen")

    print(f"::: Finished generating backgrounds")

    generation_dir = f"{local_temp_dir}/test"
    generated_bg_images_path = ["/tmp/generations/14/generated_background_0.png"
                                , "/tmp/generations/14/generated_background_1.png"]

    print(f"::: Saving the background images")

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

    x = (image_width - text_width) * 0.7
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

    x = (image_width - text_width) + 20
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
    

    


if __name__ == "__main__":
    generate_ad_banners()