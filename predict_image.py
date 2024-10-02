import os
import supervision as sv
from PIL import Image
import numpy as np
import cv2

import pycocotools.mask as mask_util
from groundingdino.util.inference import load_model, load_image, predict, annotate

HOME = "/source"
CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
detection_model = load_model(CONFIG_PATH, WEIGHTS_PATH)

IMAGE_PATH = "/source/data/dog.jpeg"

TEXT_PROMPT = "bag"  
BOX_TRESHOLD = 0.3  
TEXT_TRESHOLD = 0.3  

image_source, image = load_image(IMAGE_PATH)
print(image_source.shape, image.shape)
boxes, logits, phrases = predict(
    model=detection_model, 
    image=image, 
    caption=TEXT_PROMPT, 
    box_threshold=BOX_TRESHOLD, 
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("test.png", annotated_frame)
# %matplotlib inline  
# sv.plot_image(annotated_frame, (16, 16))