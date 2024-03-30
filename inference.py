import os

import tensorflow as tf

from utils.processor import PostProcessor
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import argparse
import glob
import numpy as np

from networks import load_models
from utils.bbox import get_patch
from utils.functions import saveResultToCsv

def loadOpts():
    parser = argparse.ArgumentParser(description='Run inference on single or mutliple images')
    parser.add_argument('-i', '--input', type=str, required=True, help='The path to image') 
    parser.add_argument('-o', '--output', type=str, required=True, help='Where to save the results') 
    parser.add_argument('--use_onnx', action="store_true", default=False, help='Where to save the results') 
    return parser.parse_args()

def resize_image_short_side(image, image_short_side=736):
    height, width, _ = image.shape
    if height < width:
        new_height = image_short_side
        new_width = int(round(new_height / height * width / 32) * 32)
    else:
        new_width = image_short_side
        new_height = int(round(new_width / width * height / 32) * 32)
    return cv2.resize(image, (new_width, new_height))

def predictDetection(det_model, img_path):
    im_name = img_path.split("/")[-1]
    raw_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    boxes = det_model.predict_one_page(raw_image)
    
    return im_name, boxes

def predictRecognition(rec_model, patch):
    return rec_model.predict_one_patch(patch).strip()     

if __name__ == '__main__':
    args = loadOpts()

    ipn = args.input
    save_path = args.output
    is_file = True
    
    if not os.path.isfile(ipn):
        is_file = False
        
    if not os.path.isdir(save_path):
        raise ValueError("Save path must is a folder")
        
    os.makedirs(save_path, exist_ok=True)
    dect_result = []
    reg_result = []
    det_model, rec_model = load_models(args.use_onnx)

    # Run detection on images
    if is_file:
        name, boxes = predictDetection(det_model, ipn)
        dect_result.append({"name": name, "path": ipn, "boxes": boxes})
        print("[Detection] found total {} boxes on file: {}".format(len(boxes), name))
    else:
        files = glob.glob(os.path.join(ipn, "*.jpg"))
        for file in files:
            name, boxes = predictDetection(det_model, file)
            dect_result.append({"name": name, "path": file, "boxes": boxes}) 
            print("[Detection] found total {} boxes on file: {}".format(len(boxes), name))

    # Run recognition on the boxes
    for idx, row in enumerate(dect_result):
        print("[Recognition] running on file: {}".format(row['name']))
        raw_image = cv2.cvtColor(cv2.imread(row['path']), cv2.COLOR_BGR2RGB)
        
        for idx2, box in enumerate(row['boxes']):
            patch = get_patch(raw_image, box)
            nom_text = predictRecognition(rec_model, patch)
            
            x1 = box[0][0]
            y1 = box[0][1]
            x2 = box[1][0]
            y2 = box[1][1]
            x3 = box[2][0]
            y3 = box[2][1]
            x4 = box[3][0]
            y4 = box[3][1]
            
            reg_result.append({'name': row['name'], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4, 'nom_text': nom_text})
            print("[Recognition] nom text at box {} is: {}".format(idx2 + 1, nom_text))
            
    saveResultToCsv(reg_result, os.path.join(save_path, "output.csv"))    
    print("[SUCCESS] saved the result to output.csv")    