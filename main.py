import os
from flask import Flask
from flask import request

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import requests
import tensorflow as tf


app = Flask(__name__)

model = tensorflow.keras.models.load_model('keras_model.h5')

@app.route("/")
def hello_world():
    name = os.environ.get("NAME", "World")
    return "Hello {}!".format(name)

@app.route("/prediction", methods=['POST'])
def prediction():
    # URL --> Image our keras model can proccess.
    def read_tensor_from_image_url(url,
                                   input_height=224,
                                   input_width=224,
                                   input_mean=0,
                                   input_std=255):
        image_reader = tf.image.decode_jpeg(
            requests.get(url).content, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize(
                    dims_expander,
                    [input_height,input_width],
                    method='bilinear',
                    antialias=True,
                    name = None
                    )
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        return normalized 

    #Get all the values in your POST request. 
    apikey = request.args.get('apikey')
    image = request.args.get('url')

    #Check for API key access  --> Very makeshift manual solution. Totally not fit for production levels.
    if apikey == 'f69c02cc-5423-4285-9993-b42ecdec1c74':  

        #Follow all the neccessary steps to get the prediction of your image. 
        image = read_tensor_from_image_url(image)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        prediction = model.predict(data)
        
        #Return the prediction and a 200 status
        return '{}'.format(prediction) , 200
    
    return 'Invalid apikey',500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
