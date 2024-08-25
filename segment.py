import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np

from groundingdino.util.inference import load_model
from groundingdino.util.inference import load_image
from groundingdino.util.inference import predict, annotate
from groundingdino.util import box_ops
from segment_anything import build_sam, SamPredictor

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
        image_source_pil = Image.fromarray(image_source)
        image_mask_pil = Image.fromarray(mask)
        composite_image = Image.new("RGBA", image_source_pil.size)
        composite_image = Image.composite(image_source_pil, composite_image, image_mask_pil)
        
        return composite_image
    
    
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


def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(BytesIO(r.content)) as im:
        im.save(image_file_path)
    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))


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
     
    
