import cv2
import numpy as np
from PIL import Image



import pytesseract
#pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\rsmm2\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'




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
            #cv2.imshow('origibal_img',origibal_img)
            license_img = origibal_img[y:y+h,x:x+w]
            text = pytesseract.image_to_string(license_img,lang='eng')
            area = w*h
            #print('area : ',area)
            x = text.split("-")
            if x is None:
                continue
            else:
                if len(x)==5:
                    #cv2.imwrite('D:/Project/ImageNew/Test_gray4.jpg', license_img) 
                    cv2.drawContours(img,[contour],-1,(255,0,255),3)
                    dim=(600,800)
                    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    #cv2.imshow('contours',img)
                    return text
                else:
                    continue
        else:
            count+=1
            continue

filename='im.jpg'
img = cv2.imread(filename)
dim=(600,800)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
res=shadowExtraction(img)
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
th2 = cv2.threshold(gray,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
#cv2.imshow('res ',th2*255)
img255=th2*255
img_contours=contours(img255,img)
img_contours=img_contours.split()
print(img_contours[1])
#f = open('employees.txt', 'w')
#f.write(str(img_contours[0]))
#f.close()

#cv2.waitKey(0)
#cv2.destroyAllWindows()

