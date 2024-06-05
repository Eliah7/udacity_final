from flask import Flask, request, render_template, redirect, url_for
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 
from extract_bottleneck_features import *
from tqdm import tqdm
from sklearn.datasets import load_files       
from keras.utils import to_categorical
import numpy as np
from glob import glob
import pickle

import cv2                
import matplotlib.pyplot as plt   
import numpy as np
import os

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def ResNet50_predict_labels(img):
    ResNet50_model = ResNet50(weights='imagenet')
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def load_VGG19():
    bottleneck_features = np.load('../bottleneck_features/DogVGG19Data.npz')
    train_VGG19 = bottleneck_features['train']
    valid_VGG19 = bottleneck_features['valid']
    test_VGG19 = bottleneck_features['test']

    VGG19_model = Sequential()
    VGG19_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
    VGG19_model.add(Dense(133, activation='softmax'))

    VGG19_model.load_weights('../saved_models/weights.best.VGG19.keras')
    print(VGG19_model.summary())

    return VGG19_model

def path_to_tensor(img):
    try:
        x = image.img_to_array(img)
        # Normalize the image tensor
        return np.expand_dims(x, axis=0).astype('float32')/255
    except IOError:
        print(f"Warning: Skipping corrupted image")
        return None
    
def VGG19_predict_breed(img):
    bottleneck_feature = extract_VGG19(path_to_tensor(img))
    predicted_vector = model.predict(bottleneck_feature)
    dog_path_name = dog_names[np.argmax(predicted_vector)]
    _, dog_breed = dog_path_name.split(".")
    return dog_breed

app = Flask(__name__)
model = load_VGG19()
dog_names = pickle.load("../dog_names")

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    decoded_preds = VGG19_predict_breed(img)
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array = preprocess_input(img_array)

    # preds = model.predict(img_array)
    # decoded_preds = decode_predictions(preds, top=3)[0]
    return decoded_preds


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            preds = classify_image(file_path)
            return render_template('result.html', preds=preds)
    return render_template('index.html')




# Print statistics about the dataset
print(f'There are {len(dog_names)} total dog categories.')


if __name__ == '__main__':
    app.run(debug=True)
