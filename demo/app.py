import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
from torchvision import transforms
from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO('best.pt')

# Définir la transformation des images
transform = transforms.Compose([transforms.Resize((640, 640)),
                                transforms.ToTensor()])

# Fonction pour prédire et afficher les résultats
def predict_and_display(image):
    # Convertir l'image en tensor
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Faire la prédiction
    results = model(image_tensor)

    # Récupérer la première prédiction
    pred = results[0]

    # Afficher l'image d'origine
    st.image(image)

    # Afficher l'image avec les prédictions
    img_pred = pred.plot()
    img_pred = np.array(img_pred)
    if len(img_pred.shape) == 2:
        # L'image est en niveaux de gris, la convertir en couleurs
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_GRAY2BGR)
    img_pred = Image.fromarray(img_pred.astype(np.uint8))
    buffered = io.BytesIO()
    img_pred.save(buffered, format="JPEG")
    buffered.seek(0)
    st.image(Image.open(buffered))

# Page principale de l'application
st.title('Application de détection d\'objets avec YOLOv8')

# Télécharger une image
uploaded_file = st.file_uploader("Télécharger une image", type="jpg")

if uploaded_file is not None:
    # Ouvrir l'image téléchargée
    image = Image.open(uploaded_file)

    # Redimensionner l'image
    image = image.resize((640, 640))

    # Faire la prédiction et afficher les résultats
    predict_and_display(image)
else:
    st.write("Veuillez télécharger une image pour effectuer une prédiction.")