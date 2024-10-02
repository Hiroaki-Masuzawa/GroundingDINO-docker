import os
import supervision as sv
from PIL import Image
import numpy as np
import cv2

import pycocotools.mask as mask_util
from groundingdino.util.inference import load_model, load_image, predict, annotate

import groundingdino.datasets.transforms as T

TEXT_PROMPT = "bottle, face, display"  
BOX_TRESHOLD = 0.3  
TEXT_TRESHOLD = 0.3  

if __name__ == '__main__':
    HOME = "/source"
    CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
    WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
    print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
    detection_model = load_model(CONFIG_PATH, WEIGHTS_PATH)

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # camera cap
    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        cv2.imshow('raw_frame', frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = Image.fromarray(frame_rgb)
        input_frame, _ = transform(input_frame, None)
        boxes, logits, phrases = predict(
            model=detection_model, 
            image=input_frame, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD
        )
        annotated_frame = annotate(image_source=frame_rgb, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imshow('result', annotated_frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
            exit()