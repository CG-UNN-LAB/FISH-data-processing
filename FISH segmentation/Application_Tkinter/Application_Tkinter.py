import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from tkinter import filedialog
from PIL import Image
import os
import czifile
import numpy as np
import json
import cv2
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from ultralytics import YOLO
#from IPython.display import display, Image
from roboflow import Roboflow


HOME = os.getcwd()
print(HOME)


from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

# Изменения размеров окна, где отображается окно
label_width = 450
label_height = 450

root = tk.Tk()

res = '..\\Application_Tkinter\\Res_for_application'
file_zag=os.path.join(res,'zag.png')
file_zag2 =os.path.join(res,'zag2.png')

imgicon = PhotoImage(file=os.path.join(res,'ico.png'))
root.tk.call('wm', 'iconphoto', root._w, imgicon) 
root.geometry("1000x600")  # Set a fixed size for the application window

image = Image.open(file_zag)
# resize the image to the desired dimensions
image = image.resize((label_width, label_height))
# convert the PIL Image to a Tkinter PhotoImage object
photo = ImageTk.PhotoImage(image)

image2 = Image.open(file_zag2)
# resize the image to the desired dimensions
image2 = image2.resize((label_width, label_height))
# convert the PIL Image to a Tkinter PhotoImage object
photo2 = ImageTk.PhotoImage(image2)

IMAGE_PREDICT = "//"
FILE = "//"
IMAGE = Image

def process_image(filename):
    if filename.endswith('.czi'):
        # Open the CZI file using czifile
        with czifile.CziFile(os.path.join(filename)) as czi:
            # Get all channels in the file
            channels = czi.asarray()[0, 0, :, :, :]
            # Combine all channels into a single RGB image
            image_array = np.stack([channels[1], channels[2], channels[0]], axis=-1)
            # Convert the image to a standard data type
            image_array = np.squeeze(image_array)
            image = np.uint8(image_array)
            img = Image.fromarray(image)
            img = img.resize((label_width, label_height))
            #добавить изображение сюда
            global IMAGE
            global IMAGE_PREDICT
            IMAGE = img
            #загрузка изображения 
            photo = ImageTk.PhotoImage(img)
            #Add image to the Canvas Items
            label.configure(image=photo)
            label.image = photo
            label.pack()

def select_image():
    filename = filedialog.askopenfilename()
    print(filename)
    global FILE
    FILE = filename
    process_image(filename)

def save_image():
    savepath = tk.filedialog.asksaveasfilename(defaultextension=".jpg")
    #save the image to the selected location and filename
    global FILE
    global IMAGE_PREDICT
    filename = FILE
    
    with czifile.CziFile(os.path.join(filename)) as czi:
            # Get all channels in the file
            channels = czi.asarray()[0, 0, :, :, :]

            # Combine all channels into a single RGB image
            image_array = np.stack([channels[1], channels[2], channels[0]], axis=-1)
            # Convert the image to a standard data type
            image_array = np.squeeze(image_array)
            image = np.uint8(image_array)
            img = Image.fromarray(image)
            img = img.resize((label_width, label_height))
            #добавить изображение сюда
            global IMAGE
            global IMAGE_PREDICT
            IMAGE = img
            #загрузка изображения 
    pil_image = img
    pil_image.save(os.path.join(savepath))
    IMAGE_PREDICT=os.path.join(savepath)

def predict_image(filename,s):
    model = YOLO('..\\Model\\my_yolov8_model_core_segmentation.pt')

    results = model.predict(source=filename, show = True, classes=[0,1],project='..\\Photo_Tkinter', save = True, hide_labels = False,hide_conf = False,conf=float(s),save_txt = True,line_thickness=1)
    names = model.names
    number_whole = 0
    number_explode = 0
    for r in results:
        for c in r.boxes.cls:
            if (names[int(c)]=="Whole cell"):  number_whole+=1
            if (names[int(c)]=="Explode cell"): number_explode+=1
    print("Explode:")
    print(number_explode)
    print("Whole:")
    print(number_whole)
    #print(results)

def select_image2():
    filename = filedialog.askopenfilename()
    print(filename)
    global FILE
    FILE = filename
    s = text.get(1.0, END)
    #label['text'] = s
    predict_image(filename,s)


f_left = Frame(root)
f_right = Frame(root)

f_left = LabelFrame(text="CZI_TO_JPEG")
f_right = LabelFrame(text="PREDICT_YOUR_IMAGE")

f_left.pack(side=tk.LEFT)
f_right.pack(side=tk.RIGHT)

global label
label = tk.Label(f_left, image=photo, width=label_width, height=label_height)
label.image = photo
label.pack(side=tk.TOP)

global label2
#label2 = tk.Label(f_right, image=photo2, width=label_width, height=label_height)
#label2.image = photo2
#label2.pack(side=tk.TOP)

label3 = tk.Label(f_right,text="ПОРОГ ТОЧНОСТИ:")
label3.pack(side=tk.LEFT)

text = Text(f_right,width=30, height=1)
text.pack(side=tk.LEFT)

button2 = tk.Button(f_left, text="Save your convert image", command=save_image)
#button2.pack(side="right", anchor="nw")  # Place the button on the left border of the window
button2.pack(side=tk.BOTTOM)

button = tk.Button(f_left, text="Select Image for image translation", command=select_image)
#button.pack(side="right", anchor="nw")  # Place the button on the left border of the window
button.pack(side=tk.BOTTOM)

#button4 = tk.Button(f_right, text="Save your predict with mask image", command=predict_image)
#button4.pack(side="right", anchor="nw")  # Place the button on the left border of the window
#button4.pack(side=tk.BOTTOM)

button3 = tk.Button(f_right, text="Predict your convert image", command=select_image2)
#button3.pack(side="right", anchor="nw")  # Place the button on the left border of the window
button3.pack(side=tk.BOTTOM)

root.mainloop()