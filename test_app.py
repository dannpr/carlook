import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
import streamlit as st
import pandas as pd
import re
import matplotlib.colors as colorsHTML




# Constants
CLASS_NAMES = ['_background_', 'back_bumper', 'back_glass', 'back_left_door','back_left_light',
               'back_right_door', 'back_right_light', 'front_bumper','front_glass',
               'front_left_door',  'front_left_light', 'front_right_door',  'front_right_light',  
               'hood',  'left_mirror', 'right_mirror', 'tailgate',  'trunk', 'wheel']
BACKBONE = 'resnet18'
BATCH_SIZE = 4
LR = 0.0001
EPOCHS = 40
n_classes = 19
activation = 'softmax'
colors = [(245,255,250), (75,0,130), (0,255,0), (32,178,170),(0,0,255), (0,255,255), (255,0,255), (128,0,128), (255,140,0),
      (85,107,47), (102,205,170), (0,191,255), (255,0,0), (255,228,196), (205,133,63),
      (220,20,60), (255,69,0), (143,188,143), (255,255,0)]

def preprocess_image(path_img):
    img = Image.open(path_img)
    ww = 512
    hh = 512
    img.thumbnail((hh, ww))
    i = np.array(img)
    ht, wd, cc= i.shape

    # create new image of desired size and color (blue) for padding
    color = (0,0,0)
    result = np.full((hh,ww,cc), color, dtype=np.uint8)

    # copy img image into center of result image
    result[:ht, :wd] = img
    return result, ht, wd

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    #plt.show()
    # Return figure
    return plt.gcf()

def denormalize(x):
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def get_legends(class_names, colors, tags):
    n_classes = len(class_names)
    print("n_classes : ",n_classes)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3), dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes], colors[:n_classes]))
    j = 0
    for (i, (class_name, color)) in class_names_colors:
        if i in tags:
          color = [int(c) for c in color]
          cv2.putText(legend, class_name, (5, (j * 25) + 17), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 0, 0), 1)
          cv2.rectangle(legend, (100, (j * 25)), (125, (j * 25) + 25), tuple(color), -1)
          j +=1

    return legend

def get_colored_segmentation_image(seg_arr, n_classes, colors=colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img/255

def concat_legends(seg_img, legend_img):
    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]
    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img

def display_original_image(img):

  # Create matplotlib figure
  fig, ax = plt.subplots() 
  ax.imshow(img)
  
  # Display in Streamlit 
  st.write("Original Image")
  st.pyplot(fig)


def predict_and_display(model, img_path):

  n = 1
  for i in range(n):

      image = np.expand_dims(preprocess_image(img_path)[0], axis=0)

      pr_mask = model.predict(image).squeeze()

      pr_mask_gray = np.zeros((pr_mask.shape[0],pr_mask.shape[1]))
      for ii in range(pr_mask.shape[2]):
          pr_mask_gray = pr_mask_gray + 1/pr_mask.shape[2]*ii*pr_mask[:,:,ii]

  # Create visualization
  fig = visualize(
    image=denormalize(image.squeeze()),
    pr_mask=pr_mask_gray
  )
  
  st.write("Prediction")
  st.pyplot(fig)


def predict_and_visualize(model, img_path):

  img = Image.open(img_path)
  tags = []
  print(img.size)
  img_scaled_arr = preprocess_image(img_path) 
  print(img.size)
  image = np.expand_dims(img_scaled_arr[0], axis=0)
  pr_mask = model.predict(image).squeeze()
  pr_mask_int = np.zeros((pr_mask.shape[0],pr_mask.shape[1]))
  kernel = np.ones((5, 5), 'uint8')
  for i in range(1,19):
    array_one = np.round(pr_mask[:,:,i])
    op = cv2.morphologyEx(array_one, cv2.MORPH_OPEN, kernel)
    if sum(sum(op ==1)) > 100:
      tags.append(i)
      pr_mask_int[op ==1] = i

  img_segmented = np.array(Image.fromarray(pr_mask_int[:img_scaled_arr[1], :img_scaled_arr[2]]).resize(img.size))  

  seg = get_colored_segmentation_image(img_segmented, 19, colors=colors)

  fused_img = ((np.array(img)/255)/2 + seg/2).astype('float32')

  seg = Image.fromarray((seg*255).astype(np.uint8))
  fused_img = Image.fromarray((fused_img*255).astype(np.uint8))
  
  plt.imshow(seg)
  plt.imshow(fused_img)

  # Display in Streamlit
  st.pyplot()

  # Add this:

  # Get legends
  legend_predicted = get_legends(CLASS_NAMES, colors, tags)

  # Concatenate legends and fused image
  final_img = concat_legends(np.array(fused_img), np.array(legend_predicted))

  # Display
  plt.figure(figsize=(20,10))
  plt.imshow(final_img)

  # Display all in Streamlit
  st.pyplot()


def main():
    img_path = 'test_image.jpeg'

    # Preprocess the image
    img_scaled_arr, ht, wd = preprocess_image(img_path)
    
    #Display original image
    display_original_image(img_scaled_arr)
    
    # Initialize the model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    
    # Load model weights
    model.load_weights('model_weights.h5')
    
    #predict &  display
    predict_and_display(model, img_path)

    #predict & visualize
    predict_and_visualize(model, img_path)
    

if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()
