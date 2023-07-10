# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:47:02 2023

@author: RANGER
"""

import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the reference image
ref_image = cv2.imread('WINPro.jpg')
ref_image = cv2.resize(ref_image, (800,800))
ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

# List of image file paths
image_files = ['2.jpeg', 'snakeandladder.jpg', 'arshad.jpg','WINPro.jpg']

# Iterate over the image files
for file in image_files:
    # Read the image
    image = cv2.imread(file)
    image = cv2.resize(image, (800,800))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (face) from the image
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize the reference face to match the size of the detected face
        ref_face_resized = cv2.resize(ref_gray, (w, h))
        
        # Calculate the absolute difference between the reference face and the detected face
        diff = cv2.absdiff(face_roi, ref_face_resized)
        
        # Calculate the mean squared error (MSE) as the difference score
        mse = np.mean(diff)
        
        # Set a threshold for matching
        threshold = 50
        
        # Calculate the match percentage
        match_percentage = (1 - (mse / threshold)) * 100
        match_percentage = max(match_percentage, 0)
        
        # Draw a rectangle around the detected face and display the match percentage
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f"Match: {match_percentage:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the image with detected faces and match percentage
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
