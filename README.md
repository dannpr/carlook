# Carlook

## Dataset of car  used

- [The Comprehensive Cars (CompCars) dataset](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
- [https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt)

- [https://drive.google.com/drive/folders/18EunmjOJsbE5Lh9zA0cZ4wKV6Um46dkg](https://drive.google.com/drive/folders/18EunmjOJsbE5Lh9zA0cZ4wKV6Um46dkg)

## Create your model with google collab

- [carlook_training_segmentation_YOLOv8](https://colab.research.google.com/drive/1eloee7PP6GV3PXfaoVW6Ch20eOO2WhSi?usp=sharing)

## Pre-requisites
- Python and pyvenv (or Python3)

## Installation - Steps  
1. Create Python virtual environment
    ```
    python3 -m venv {name of your venv}
    ```
    Or
    ```
    python -m venv {name of your venv}
    ```
    
2. Activate virtual environment
    ```
    source {name of your venv}/bin/activate
    ```
    
You should see the name of your environment on the left of your terminal

3. Go the demo folder 
    ```
    cd demo
    ```

4. Install requirements 

Be sure to launch under your python environment
You should see the name of your environment on the left of your terminal
    ```
    pip install -r requirements.txt
    ```

5. Launch the app

Be sure to launch under your python environment
You should see the name of your environment on the left of your terminal
    ``` 
    streamlit run app.py
    ```

## Overview

Carlook is an innovative application designed to revolutionize the automobile industry. It enables users to effortlessly identify various car parts using image recognition technology and seamlessly integrates with marketplaces for automobile parts procurement.

### The Technology

Carlook stands out as the first app to offer a car part analysis tool. Mechanics or users can simply snap a photo of a vehicle, and the integrated AI promptly analyzes and identifies the parts visible in the image. This technology is geared towards automating the detection of car components, streamlining the repair and replacement process.

### Key Features

---

- **Advanced Object Detection**: Utilizes state-of-the-art image recognition to identify car parts from photographs.
- **User-Friendly Interface**: Simplified and intuitive design for easy navigation and usage.

### Upcoming Enhancements

---

- **Real-Time Object Detection**: Expanding capabilities to include live recognition of any car part using video input or camera roll.
- **Automated Photographic Capture**: Implementing real-time detection for immediate and automated photo capture.

### Usage

- **Model Testing & Extraction**: Leveraging Google Colab for rigorous testing of models and their subsequent extraction for practical application.

As demonstrated in Version 1:

1. **Photo Upload**: Users can start by uploading a photo of their vehicle. See an example of the upload interface here: ![File Upload](./assets/filedropped.png)

2. **Color-Coded Results**: Users can then see the results with color-coded annotations, indicating the specific car parts identified by the AI. Here's how the results are displayed: ![Color-Coded Results](./assets/result.png)


## Licensing

Carlook is released under the MIT License, ensuring open-source accessibility and development potential.

## Get in Touch

For inquiries or feedback regarding Carlook, feel free to reach out at [contact@polyblocks.xyz](mailto:contact@polyblocks.xyz).
