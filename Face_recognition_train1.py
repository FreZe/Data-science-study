#!/usr/bin/env python
# coding: utf-8




import cv2
import face_recognition

video_capture = cv2.VideoCapture(1)
face_locations = []

while True:
     ret, frame = video_capture.read()
# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
     rgb_frame = frame
# Find all the faces in the current frame of video
     face_locations = face_recognition.face_locations(rgb_frame)
# Display the results
     for top, right, bottom, left in face_locations:
 # Draw a box around the face
         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
# Display the resulting image
         cv2.imshow('Video', frame)
# Hit ‘q’ on the keyboard to quit!
         if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()

