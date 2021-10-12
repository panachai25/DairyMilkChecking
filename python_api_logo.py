from flask import Flask, jsonify, request
import json
import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

#declared an empty variable for reassignment


        
response = ''

#creating the instance of our flask application
app = Flask(__name__)

#route to entertain our post and get request from flutter app
@app.route('/name', methods = ['GET', 'POST'])
def nameRoute():

    #fetching the global response variable to manipulate inside the function
    global response
    #checking the request type we get from the app
    if(request.method == 'POST'):
        request_data = request.data #getting the response data
        request_data = json.loads(request_data.decode('utf-8')) #converting it from json to key value pair
        name = request_data['name']#assigning it to name
        Type = request_data['Type']
        print(Type)
        im=base64.b64decode(name)
        f=open('im.jpg','wb')
        f.write(im)
        filename='im.jpg'
        img = cv2.imread(filename)
        model = load_model('64x3-CNN_Logo.model')
        dim=(200,200)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out=[]
        i=0
        out=[]
        i=0
        correct=[]
        result_pred=[]
        classes=['DuchMilk','Foremost','SchoolMilk','Vitamilk']
        correct=np.zeros(len(classes),np.uint)
        incorrect=np.zeros(len(classes),np.uint)
        image=np.expand_dims(img/255.0, 0)
        result = model.predict_classes(image)
        print(result)
        result_pred=classes[int(result)]
        print(result_pred)
        response = f'Hi {name}! this is Python' #re-assigning response with the name we got from the user
        return jsonify({'name' : result_pred}) #to avoid a type error 
    else:
        return jsonify({'name' : response}) #sending data back to your frontend app

if __name__ == "__main__":
    app.run()

