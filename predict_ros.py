#!/usr/bin/env python3
## coding: UTF-8

import os
import yaml
import numpy as np 

from PIL import Image
import cv2

import rospy
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROSImage

from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T

class ROSGroundingDINO:
    def __init__(self, ):
        HOME = "/source"
        CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
        WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
        print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
        self.detection_model = load_model(CONFIG_PATH, WEIGHTS_PATH)
        self.text_pronpt = "bag"  
        self.box_treshold = 0.3  
        self.text_treshold = 0.3
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        rospy.init_node('graundingdino', anonymous=True)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/usb_cam/image_raw', ROSImage, self.image_sub)
        self.pub_resultstr = rospy.Publisher('result_str', String, queue_size=1)
        self.pub_dbgimg = rospy.Publisher('bboximg', ROSImage, queue_size=1)

    def image_sub(self, msg):
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width  = frame_bgr.shape[0:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_frame = Image.fromarray(frame_rgb)
        input_frame, _ = self.transform(input_frame, None)
        boxes, logits, phrases = predict(
            model=self.detection_model, 
            image=input_frame, 
            caption=self.text_pronpt, 
            box_threshold=self.box_treshold, 
            text_threshold=self.text_treshold
        )
        
        annotated_frame = annotate(image_source=frame_rgb, boxes=boxes, logits=logits, phrases=phrases)
        ret_dict = {}
        ret_dict['header'] = {'seq':msg.header.seq, 'stamp': msg.header.stamp.to_sec(), 'frame_id': msg.header.frame_id}
        ret_dict['boxes'] = [np.array([box[0]*width, box[1]*height, box[2]*width, box[3]*height,]).tolist() for box in np.array(boxes)]
        ret_dict['scores'] = [float(score) for score in logits]
        ret_dict['names'] =  [phrase for phrase in phrases]
        ret_str = yaml.dump(ret_dict)

        if self.pub_resultstr.get_num_connections() > 0:
            self.pub_resultstr.publish(String(ret_str))
        if self.pub_dbgimg.get_num_connections() > 0:
            dbgimg_msg = self.bridge.cv2_to_imgmsg(annotated_frame, 'bgr8')
            dbgimg_msg.header = msg.header
            self.pub_dbgimg.publish(dbgimg_msg)
        

if __name__ == '__main__':
    hoge  = ROSGroundingDINO()
    rospy.spin()