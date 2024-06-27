import pipeline
import pickle
from config import Config
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image,ImageEnhance, ImageOps
import random
from tqdm import tqdm
import cv2

def random_transformation(img):
    transform_type = random.choice(['rotate', 'zoom', 'enhance'])
    
    if transform_type == 'rotate':
        # Apply random rotation
        angle = random.uniform(-15, 15)  # Rotate between -15 to 15 degrees
        img = img.rotate(angle)
    
    elif transform_type == 'zoom':
        scale = random.uniform(0.8, 1.2)  # Zoom between 80% to 120%
        width, height = img.size
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        img = img.resize((new_width, new_height))
        
        # If zoomed in (larger than original), crop back to original size
        if scale > 1:
            img = img.crop((0, 0, width, height))
        else:  # If zoomed out (smaller than original), pad to original size
            img = ImageOps.pad(img, (width, height))
    
    elif transform_type == 'enhance':
        # Apply random enhancement (brightness, contrast, etc.)
        enhancer_type = random.choice(['brightness', 'contrast', 'sharpness'])
        if enhancer_type == 'brightness':
            enhancer = ImageEnhance.Brightness(img)
        elif enhancer_type == 'contrast':
            enhancer = ImageEnhance.Contrast(img)
        else:
            enhancer = ImageEnhance.Sharpness(img)
        factor = random.uniform(0.8, 1.2)  # Enhance by a factor between 0.8 to 1.2
        img = enhancer.enhance(factor)
    
    return (img,transform_type)

def save_random_samples(images, num_samples, output_dir):
    sampled_images = []
    for i in range(num_samples):
        num = random.randrange(0,len(images))
        image = images[num,:,:,0]
        image = cv2.bitwise_not(image)
        sampled_images.append(image)
    os.makedirs(output_dir, exist_ok=True)

    for i, img_array in enumerate(tqdm(sampled_images, desc="Saving images")):
        img = Image.fromarray(img_array)
        img,transformation = random_transformation(img)
        img.save(os.path.join(output_dir, f"sample_{i}_{transformation}.png"))

    print("Sampled images have been saved successfully.")

output_dir = 'D:\year-end-project\models\gui\prints'

dataset = pipeline.CreateDataset('Kmnist')
dataset.run()
(X_train, y_train), (X_test , y_test)= dataset.get_train_test()

save_random_samples(X_train,100,output_dir)