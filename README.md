# Medical Image Augmentation and Synthesis

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Training the Model](#training-the-model)
- [Using the GUI](#using-the-gui)
- [Acknowledgements](#acknowledgements)

## Overview

This project focuses on generating synthetic medical images using Generative Adversarial Networks (GANs). The primary goal is to augment the dataset of brain tumor images to improve the performance of machine learning models for medical image analysis.


## Technologies Used

- **Programming Language:** Python
- **Libraries and Frameworks:**
  - TensorFlow
  - Keras
  - Numpy
  - Pillow (PIL)
  - Tkinter
- **Hardware:**
  - Nvidia GeForce RTX 4090 GPU

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xyntrae/medical_image_synthesis.git
   cd medical_image_synthesis

2. **Install the required dependencies:**
    pip install -r requirements.txt

3. **Ensure you have the required datasets in the correct folder structure:**
    -Place the brain tumor images in data/brain_tumor/.
    -Place the non-brain tumor images in data/non_brain_tumor/.


## Training the Model

1. **Preprocess the data:**
   python data_preprocessing.py

2. **Train the GAN model:**
    python model_training.py
    -This will train the GAN and save the weights of the generator and discriminator models.

## Using the GUI

1. **Run the GUI application:**
   python gui_app.py

## Acknowledgements
This project utilizes various open-source libraries and frameworks, without which this implementation would not be possible. Special thanks to the TensorFlow and Keras teams for their continued efforts in providing excellent tools for machine learning and deep learning.
