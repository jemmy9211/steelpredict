# File directory:
# model_final.pth is to be put separately for each type of inspection, e.g. ../hook_angle/model_final.pth

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

APP_ROOT = os.path.abspath(os.path.dirname(__file__))

def inference(model_type, image_dir):
    if model_type == "h":
        model_dir = os.path.join(APP_ROOT, "hook_angle")
    elif model_type == "o":
        model_dir == os.path.join(APP_ROOT, "overlap")
    elif model_type == "s":
        model_dir == os.path.join(APP_ROOT, "spacing")
    else:
        print("Error: Unknown model type")
        return None

    im = cv2.imread(image_dir)
    print("Image received!")

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library:
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # Use `Visualizer` to draw the predictions on the image
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])