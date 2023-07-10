# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:13:43 2023

@author: RANGER
"""

import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the reference image
ref_image = cv2.imread('WINPro.jpg')
ref_image = cv2.resize(ref_image,(900,900))
ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame from the camera feed
    ret, frame = camera.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection on the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (face) from the frame
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
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the match percentage
        cv2.putText(frame, f"Match: {match_percentage:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Face Matching', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
