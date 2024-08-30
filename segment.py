import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from sklearn.cluster import KMeans
from skimage import color
from scipy.spatial.distance import euclidean

from groundingdino.util.inference import load_model
from groundingdino.util.inference import load_image
from groundingdino.util.inference import predict, annotate
from groundingdino.util import box_ops
from segment_anything import build_sam, SamPredictor
from download_image import download_image

class GroundedSAMComposite():
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dino_model = load_model(
            model_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path="GroundingDINO/grouding_dino_weights.pth"
        )
        sam_checkpoint = 'sam_vit_h_4b8939.pth'
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(self.device))
        
    
    def predict(self, local_image_path: str, prompt: str):
        # Load image
        image_source, image = load_image(local_image_path)
        
        # GroundingDino predict
        annotated_frame, detected_boxes, logits, phrases = self._dino_detect(image, image_source, prompt)
    
        # Segment Anything        
        segmented_frame_masks = self._segment(image_source, self.sam_predictor, boxes=detected_boxes)
        mask = segmented_frame_masks[0][0].cpu().numpy()

        # Load images with 
        image_source_pil = Image.fromarray(image_source)
        image_mask_pil = Image.fromarray(mask)

        image_source_width, image_source_height = image_source_pil.size

        is_centered = self._check_product_item_centered(
            image_height=image_source_height,
            image_width=image_source_width,
            boxes=detected_boxes
        )

        is_background_uniform = self._check_background_uniformity(
           segmented_frame_masks=segmented_frame_masks,
           image_source_pil=image_source_pil 
        )

        is_sufficient_contrast = self._check_color_analysis(
            image_source_pil=image_source_pil
        )

        composite_image = Image.new("RGBA", image_source_pil.size)
        composite_image = Image.composite(image_source_pil, composite_image, image_mask_pil)

        return is_centered, is_background_uniform, is_sufficient_contrast, composite_image
    
    
    def _dino_detect(self, image, image_source, prompt):
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image,
            caption=prompt,
            box_threshold=0.3,
            text_threshold=0.25
        )
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        return annotated_frame, boxes, logits, phrases


    def _segment(self, image, sam_model, boxes):
        sam_model.set_image(image)
        H, W, _ = image.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image.shape[:2])
        masks, _, _ = sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
            )
        return masks.cpu()


    def _draw_mask(self, mask, image, random_color=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


    def _create_composite_image(segmented_frame_masks, image_source):
        mask = segmented_frame_masks[0][0].cpu().numpy()
        image_source_pil = Image.fromarray(image_source)
        image_mask_pil = Image.fromarray(mask)
        
        composite_image = Image.new("RGBA", image_source_pil.size)
        composite_image = Image.composite(image_source_pil, composite_image, image_mask_pil)
        
        return composite_image
    

    def _check_product_item_centered(self, boxes, image_width, image_height):
        if len(boxes) == 1:
            box = boxes[0]
            x_min_normalised, y_min_normalised, x_max_normalised, y_max_normalised = box
            x_min = x_min_normalised * image_width
            y_min = y_min_normalised * image_height
            x_max = x_max_normalised * image_width
            y_max = y_max_normalised * image_height
            
            acceptable_x_range = (image_width * 0.4, image_width * 0.6)
            acceptable_y_range = (image_height * 0.4, image_height * 0.8)

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            if not (acceptable_x_range[0] <= center_x <= acceptable_x_range[1] and acceptable_y_range[0] <= center_y <= acceptable_y_range[1]):
                return False
            else:
                return True
        else:
            return False
        
    
    def _check_background_uniformity(self, segmented_frame_masks, image_source_pil):
        try:
            mask = segmented_frame_masks[0][0].cpu().numpy()
            inverted_mask = ((1 - mask) * 255).astype(np.uint8)
            image_mask_pil = Image.fromarray(inverted_mask)
            background = Image.new("RGBA", image_source_pil.size)
            background = Image.composite(image_source_pil, background, image_mask_pil)
            gray_background = background.convert("L")
            gray_background_np = np.array(gray_background)
            variance = np.var(gray_background_np)
            uniformity_score = 1 / (1 + variance)
            scaled_uniformity_score = uniformity_score * 1e5
            threshold = 8
            return scaled_uniformity_score > threshold
        except:
            return False
        
    
    def _check_color_analysis(self, image_source_pil, threshold=20):
        image_np = np.array(image_source_pil)
        pixels = image_np.reshape(-1,3)
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_
        colors_lab = color.rgb2lab(colors[np.newaxis, :, :])[0]
        contrast_scores = []
        for i in range(len(colors_lab)):
            for j in range(i + 1, len(colors_lab)):
                # Calculate Euclidean distance between each pair of LAB colors
                contrast = euclidean(colors_lab[i], colors_lab[j])
                contrast_scores.append(contrast)
        
        sufficient_contrast = all(score >= threshold for score in contrast_scores)
        return sufficient_contrast, colors, contrast_scores 



if __name__ == "__main__":
    local_image_path = "assets/inpaint_demo.jpg"
    image_url = "https://m.media-amazon.com/images/I/61AIroGIryL._SX522_.jpg"

    download_image(image_url, local_image_path)
    
    image_source, image = load_image(local_image_path)
    instance = GroundedSAMComposite()
    
    # Grounding DINO inference
    annotated_frame, detected_boxes, logits, phrases = instance._dino_detect(image, text_prompt="toys")
    
    # Segment the object
    segmented_frame_masks = instance._segment(image_source, instance.sam_predictor, boxes=detected_boxes)
    annotated_frame_with_mask = instance._draw_mask(segmented_frame_masks[0][0], annotated_frame)
     
    
