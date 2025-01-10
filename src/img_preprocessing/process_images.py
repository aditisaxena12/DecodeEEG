import os
import cv2
import numpy as np

'''
This script is used to preprocess images for training the decoder model
'''

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images, target_size=(500,500)):
    processed_images = []
    for img in images:
        img_resized = cv2.resize(img, target_size)
        img_normalized = img_resized / 255.0
        processed_images.append(img_normalized)
    return np.array(processed_images)

def get_concept(images_folder):
    #iterate over all concept folders
    filenames= os.listdir (images_folder)
    for foldername in filenames:
        input_folder  = images_folder + foldername
        print("Processing ...",foldername)
        images = load_images_from_folder(input_folder)
        processed_images = preprocess_images(images)
        np.save("/home/aditis/decodingEEG/DecodeEEG/data/images/processed_images/training/" + foldername+'.npy', np.array(processed_images))

def main():
    folder_path = '/home/aditis/decodingEEG/DecodeEEG/data/images/training_images/'
    get_concept(folder_path)

if __name__ == "__main__":
    main()