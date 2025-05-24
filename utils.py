import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__(**kwargs)

    def call(self, anchor, validation):
        return tf.abs(anchor - validation)

# Load model
siamese_model = tf.keras.models.load_model(
    'weights/siamese_model.h5', 
    custom_objects={'L1Dist': L1Dist}
)

image_size = (105, 105)

def preprocessingImage(image):
    image = cv2.resize(image, image_size)
    image = image / 255.0
    image = image.reshape((1, 105, 105, 3))
    return image

def getSimilarImages(image,type):
    image_path = "database/images/"   #Path of similar images
    image_info = "database/info/"    #Info of similar images(store/price)
    anc_image = preprocessingImage(image)
    similar_images = []                #contains similar images and their info
    
    for img in os.listdir(image_path + type):
        ver_image = cv2.imread(image_path + type + "/" + img)
        ver_image_edited = preprocessingImage(ver_image)
        y_hat = siamese_model.predict([anc_image,ver_image_edited])
        if y_hat[0,0] >= 0.7:
            file_name_without_extension = os.path.splitext(img)[0]
            file_path = image_info + type + "/" + file_name_without_extension + ".txt"
            with open(file_path, 'r') as file:
                # Read the entire content of the file into a string
                file_content = file.read()
            similar_images.append([ver_image,file_content])
    return similar_images