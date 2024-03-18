import streamlit as st
import glob
from ultralytics import YOLO
import cv2
from utils.utils import get_simplified_annotation_for_image
import pandas as pd
import numpy as np

"""
## Yolo Performance Exploration

"""    

def draw_annotations(ann):
    cmap = {"D00": (255,0,0), "D10": (0,255,0), "D20":(0,0,255), "D40": (255,0,255)}
    img = ann.orig_img.copy()
    for xyxy, c in zip(ann.boxes.xyxy, ann.boxes.cls):
        clsname = ann.names[int(c)]
        color = cmap[clsname]
        xmin, ymin, xmax, ymax = [int(n) for n in xyxy]
        print(xmin, ymin, xmax, ymax)
        img = cv2.rectangle(img,  (xmin, ymin), (xmax, ymax), color, 4)
        img = cv2.putText(img, clsname, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,3)
    return img   
    
def draw_xml_annotations(img, annotations):
    cmap = {"D00": (255,0,0), "D10": (0,255,0), "D20":(0,0,255), "D40": (255,0,255)}
    for annotation in annotations["objects"]:
        xmin = annotation["coords"]["xmin"]
        ymin = annotation["coords"]["ymin"]
        xmax = annotation["coords"]["xmax"]
        ymax = annotation["coords"]["ymax"]
        img = cv2.rectangle(img,  (xmin, ymin), (xmax, ymax), cmap[annotation["class"]], 4)
        img = cv2.putText(img, annotation["class"], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cmap[annotation["class"]],3)
    return img

def compare_model(image_path, model_path):
    model = YOLO(model_path)
    predictions = model(image_path)[0]
    sa = get_simplified_annotation_for_image(image_path)
    img1 = draw_annotations(predictions)
    img2 = draw_xml_annotations(predictions.orig_img.copy(), sa)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    return img1, img2


models = glob.glob("/home/daniel/repos/roaddamage/runs/**/**.pt", recursive = True)
model = st.selectbox("Model", models)

images = glob.glob("/home/daniel/repos/roaddamage/datasets/val/images/**.jpg", recursive = True)
images.sort()
count_slider = st.slider("select image", min_value = 0, max_value = len(images))

im1, im2 = compare_model(images[count_slider], model)
st.write(f"Current Image: {images[count_slider]}")
st.image([im1, im2], caption = ["Yolo Predictions", "Original Annotations"])

df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [60.39299, 5.32415],
    columns=['lat', 'lon'])
e
st.map(df, size = 1)