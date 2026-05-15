import os
import sys 
import argparse
import glob 
import time 

import cv2 
import numpy as np 
from ultralytics import YOLO

parser  = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file (test.jpg)',
                    required=True)
parser.add_argument()