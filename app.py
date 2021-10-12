from flask import Flask, jsonify, request
import json
import base64
import cv2
import numpy as np
from PIL import Image
import pytesseract
from tensorflow.keras.models import load_model
import os

#declared an empty variable for reassignment

def shadowExtraction(img):
    bgr_planes = cv2.split(img)
    res_md = []
    result_planes = []
    for plane in bgr_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        #cv2.imshow('bg_img ',bg_img)
        diff_img = 255-cv2.absdiff(plane, bg_img)
        #cv2.imshow('diff_img ',diff_img)
        res_md.append(bg_img)
        result_planes.append(diff_img)
    result = cv2.merge(result_planes)
    
    return result
def contours(img255,img):
    count=0
    img255_copy=img255.copy()
    origibal_img = img.copy()
    contours, hierarchy=cv2.findContours(img255_copy,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img,contours,-1,(0,255,255),3)
    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:8]
    
    for contour in contours:
        pre = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,0.02*pre,True)
        x,y,w,h = cv2.boundingRect(contour)
        
        if len(approx)>=4 and len(approx)<=8:
            
            license_img = origibal_img[y:y+h,x:x+w]
            custom_config = r'--oem 3 --psm 6 outputbase digits'
            text = pytesseract.image_to_string(license_img,config=custom_config)
            area = w*h
          
            x = text.split("-")
            
            if x is None:
                continue
            else:
                if len(x)==5:
          
                    if len(x[0]) > 2:
                        x_re=x[0]
                        removed = x[0].replace(x_re[0], "")
                        text=text.replace(x[0],removed)
                        
                    if len(x[len(x)-1]) > 4:
                        x_re=x[len(x)-1]
                        re = x[len(x)-1][4]
                        removed = x[len(x)-1].replace(re, "")
                        text=text.replace(x_re,removed)
                        
                   
                    return text
                else:
                    continue
        else:
            count+=1
            continue
        
response = ''


app = Flask(__name__)


@app.route('/name', methods = ['GET', 'POST'])
def nameRoute():

    
    global response
   
    if(request.method == 'POST'):
        request_data = request.data #getting the response data
        request_data = json.loads(request_data.decode('utf-8')) #converting it from json to key value pair
        name = request_data['name']#assigning it to name
        Type = request_data['Type']
        im=base64.b64decode(name)
        f=open('im.jpg','wb')
        f.write(im)
        filename='im.jpg'
        img = cv2.imread(filename)
        if Type=='fda':
            dim=(600,800)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            res=shadowExtraction(img)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            th2 = cv2.threshold(gray,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            
            img255=th2*255
            img_contours=contours(img255,img)
            img_contours=img_contours.split()
            
            output = img_contours[0]
            print(output)
            
        elif Type == 'logo':
            model = load_model('64x3-CNN_Logo.model')
            dim=(200,200)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out=[]
            out=[]
            i=0
            correct=[]
            result_pred=[]
            classes=['DuchMilk','Foremost','SchoolMilk','Vitamilk']
            correct=np.zeros(len(classes),np.uint)
            incorrect=np.zeros(len(classes),np.uint)
            image=np.expand_dims(img/255.0, 0)
            result = model.predict_classes(image)
            
            output=classes[int(result)]
            print(output)
        elif Type=='date':
            dim=(400,200)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            img_blur=cv2.medianBlur(img,3)
            kernel = np.ones((4,4),np.uint8)
            img_erode=cv2.erode(img_blur, kernel, iterations = 1)
            img_opening=cv2.morphologyEx(img_erode, cv2.MORPH_OPEN, kernel)
            custom_config = r'--oem 3 --psm 6 outputbase digits'
            text = pytesseract.image_to_string(img_opening,config=custom_config)
            output = text
        return jsonify({'name' : output}) #to avoid a type error 
    else:
        return jsonify({'name' : response}) #sending data back to your frontend app

if __name__ == "__main__":
    app.run()

