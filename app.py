from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re

from flask import *
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

import requests
import pandas as pd
import numpy as np
import pickle


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'



@app.route('/')
def home():
    return render_template('home.html')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

#model = ResNet50(weights='imagenet')
#model.save('model_resnet50.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# first run the above code , then comment it and run the below the code

# Model saved with Keras model.save()
MODEL_PATH = 'model_resnet50.h5'
# Load your saved model
model = load_model(MODEL_PATH,compile=True)
model.make_predict_function()          # Necessary
#print('Model loaded. Start serving...')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/imageclassification' , methods=['GET'])
def index():

    return render_template('icform.html')

@app.route('/imageclassification/predict' , methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string

        return result

    return None

if __name__ == "__main__":
    app.run(port=5081, debug=True)