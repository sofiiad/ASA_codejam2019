# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:54:38 2019

@author: Aymar
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 12:42:25 2019

@author: Aymar
"""

# this file is used to detect face 
# and then store the data of the face 
import cv2 
import numpy as np 
import base64
from PIL import Image
from io import BytesIO

# import the file where data is 
# stored in a csv file format 
import npwriter

def signUp(request):
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    f_list = []
    
    name = request.form['usrname']
    
    encoded1 = request.form['im1']
    encoded1 = encoded1.split(",")[1]
    unencoded1 = base64.b64decode(encoded1)
    img = Image.open(BytesIO(unencoded1))
    frame = np.asarray(img, dtype='uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
                                    reverse = True) 
    faces = faces[:1]
    if len(faces) == 1: 
        # this is removing from tuple format     
        face = faces[0] 
        
        # storing the coordinates of the 
        # face in different variables 
        x, y, w, h = face 

        # this is will show the face 
        # that is being detected     
        im_face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        f_list.append(gray_face.reshape(-1))
        
    encoded1 = request.form['im2']
    encoded1 = encoded1.split(",")[1]
    unencoded1 = base64.b64decode(encoded1)
    img = Image.open(BytesIO(unencoded1))
    frame = np.asarray(img, dtype='uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
                                    reverse = True) 
    faces = faces[:1]
    if len(faces) == 1: 
        # this is removing from tuple format     
        face = faces[0] 
        
        # storing the coordinates of the 
        # face in different variables 
        x, y, w, h = face 

        # this is will show the face 
        # that is being detected     
        im_face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        f_list.append(gray_face.reshape(-1))
    
    
    encoded1 = request.form['im3']
    encoded1 = encoded1.split(",")[1]
    unencoded1 = base64.b64decode(encoded1)
    img = Image.open(BytesIO(unencoded1))
    frame = np.asarray(img, dtype='uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
                                    reverse = True) 
    faces = faces[:1]
    if len(faces) == 1: 
        # this is removing from tuple format     
        face = faces[0] 
        
        # storing the coordinates of the 
        # face in different variables 
        x, y, w, h = face 

        # this is will show the face 
        # that is being detected     
        im_face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        f_list.append(gray_face.reshape(-1))
    
    encoded1 = request.form['im4']
    encoded1 = encoded1.split(",")[1]
    unencoded1 = base64.b64decode(encoded1)
    img = Image.open(BytesIO(unencoded1))
    frame = np.asarray(img, dtype='uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
                                    reverse = True) 
    faces = faces[:1]
    if len(faces) == 1: 
        # this is removing from tuple format     
        face = faces[0] 
        
        # storing the coordinates of the 
        # face in different variables 
        x, y, w, h = face 

        # this is will show the face 
        # that is being detected     
        im_face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        f_list.append(gray_face.reshape(-1))
    
    encoded1 = request.form['im5']
    encoded1 = encoded1.split(",")[1]
    unencoded1 = base64.b64decode(encoded1)
    img = Image.open(BytesIO(unencoded1))
    frame = np.asarray(img, dtype='uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
                                    reverse = True) 
    faces = faces[:1]
    if len(faces) == 1: 
        # this is removing from tuple format     
        face = faces[0] 
        
        # storing the coordinates of the 
        # face in different variables 
        x, y, w, h = face 

        # this is will show the face 
        # that is being detected     
        im_face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        f_list.append(gray_face.reshape(-1))
    
    encoded1 = request.form['im6']
    encoded1 = encoded1.split(",")[1]
    unencoded1 = base64.b64decode(encoded1)
    img = Image.open(BytesIO(unencoded1))
    frame = np.asarray(img, dtype='uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
                                    reverse = True) 
    faces = faces[:1]
    if len(faces) == 1: 
        # this is removing from tuple format     
        face = faces[0] 
        
        # storing the coordinates of the 
        # face in different variables 
        x, y, w, h = face 

        # this is will show the face 
        # that is being detected     
        im_face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        f_list.append(gray_face.reshape(-1))
    
    encoded1 = request.form['im7']
    encoded1 = encoded1.split(",")[1]
    unencoded1 = base64.b64decode(encoded1)
    img = Image.open(BytesIO(unencoded1))
    frame = np.asarray(img, dtype='uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
                                    reverse = True) 
    faces = faces[:1]
    if len(faces) == 1: 
        # this is removing from tuple format     
        face = faces[0] 
        
        # storing the coordinates of the 
        # face in different variables 
        x, y, w, h = face 

        # this is will show the face 
        # that is being detected     
        im_face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        f_list.append(gray_face.reshape(-1))
    
    encoded1 = request.form['im8']
    encoded1 = encoded1.split(",")[1]
    unencoded1 = base64.b64decode(encoded1)
    img = Image.open(BytesIO(unencoded1))
    frame = np.asarray(img, dtype='uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
                                    reverse = True) 
    faces = faces[:1]
    if len(faces) == 1: 
        # this is removing from tuple format     
        face = faces[0] 
        
        # storing the coordinates of the 
        # face in different variables 
        x, y, w, h = face 

        # this is will show the face 
        # that is being detected     
        im_face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        f_list.append(gray_face.reshape(-1))
    
    encoded1 = request.form['im9']
    encoded1 = encoded1.split(",")[1]
    unencoded1 = base64.b64decode(encoded1)
    img = Image.open(BytesIO(unencoded1))
    frame = np.asarray(img, dtype='uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
                                    reverse = True) 
    faces = faces[:1]
    if len(faces) == 1: 
        # this is removing from tuple format     
        face = faces[0] 
        
        # storing the coordinates of the 
        # face in different variables 
        x, y, w, h = face 

        # this is will show the face 
        # that is being detected     
        im_face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        f_list.append(gray_face.reshape(-1))
    
    encoded1 = request.form['ima']
    encoded1 = encoded1.split(",")[1]
    unencoded1 = base64.b64decode(encoded1)
    img = Image.open(BytesIO(unencoded1))
    frame = np.asarray(img, dtype='uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
                                    reverse = True) 
    faces = faces[:1]
    if len(faces) == 1: 
        # this is removing from tuple format     
        face = faces[0] 
        
        # storing the coordinates of the 
        # face in different variables 
        x, y, w, h = face 

        # this is will show the face 
        # that is being detected     
        im_face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        f_list.append(gray_face.reshape(-1))
    
    npwriter.write(name, np.array(f_list))
        
        

#name = input("Enter your name: ") 
#
## this is used to access the web-cam 
## in order to capture frames 
#cap = cv2.VideoCapture(0) 
#
#classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#"C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml") 
#
## this is class used to detect the faces as provided 
## with a haarcascade_frontalface_default.xml file as data 
#f_list = [] 
#
#while True: 
#    ret, frame = cap.read() 
#    # converting the image into gray 
#    # scale as it is easy for detection 
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    # detect multiscale, detects the face and its coordinates 
#    faces = classifier.detectMultiScale(gray, 1.5, 5) 
#    
#    # this is used to detect the face which 
#    # is closest to the web-cam on the first position 
#    faces = sorted(faces, key = lambda x: x[2]*x[3], 
#                                    reverse = True) 
#
#    # only the first detected face is used 
#    faces = faces[:1] 
#    
#    # len(faces) is the number of 
#    # faces showing in a frame 
#    if len(faces) == 1: 
#        # this is removing from tuple format     
#        face = faces[0] 
#        
#        # storing the coordinates of the 
#        # face in different variables 
#        x, y, w, h = face 
#
#        # this is will show the face 
#        # that is being detected     
#        im_face = frame[y:y + h, x:x + w]
#        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
#        gray_face = cv2.resize(gray_face, (100, 100))
#        f_list.append(gray_face.reshape(-1))
#        cv2.imshow("face", im_face)
#        if len(f_list) == 3: 
#            break
#
#
#    if not ret: 
#        continue
#
#    cv2.imshow("full", frame) 
#
#    key = cv2.waitKey(1) 
#
#    # this will break the execution of the program 
#    # on pressing 'q' and will click the frame on pressing 'c' 
#    if key & 0xFF == ord('q'): 
#        break
#    elif key & 0xFF == ord('c'): 
#        if len(faces) == 1:
#            gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY) 
#            gray_face = cv2.resize(gray_face, (100, 100)) 
#            print(len(f_list), type(gray_face), gray_face.shape) 
#
#            # this will append the face's coordinates in f_list 
#            f_list.append(gray_face.reshape(-1)) 
#        else: 
#            print("face not found",len(faces))
#        # this will store the data for detected 
#        # face 10 times in order to increase accuracy 
#        if len(f_list) == 1: 
#            break
#
## declared in npwriter 
#npwriter.write(name, np.array(f_list)) 
#
#
#cap.release() 
#cv2.destroyAllWindows() 
