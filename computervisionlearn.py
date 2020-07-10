
import face_recognition
import cv2
import numpy as np
import dlib

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

Alexandra_image = face_recognition.load_image_file('WIN_20190913_16_29_30_Pro.jpg')
Alexandra_face_encoding = face_recognition.face_encodings(Alexandra_image)[0]

Sergey_image = face_recognition.load_image_file('photo_2020-07-10_01-00-03.jpg')
Sergey_face_encoding = face_recognition.face_encodings(Sergey_image)[0]

Evgeniy_image = face_recognition.load_image_file('photo_2020-07-10_01-05-37.jpg')
Evgeniy_face_encoding = face_recognition.face_encodings(Evgeniy_image)[0]

Papa_image = face_recognition.load_image_file('photo_2020-07-10_01-28-36.jpg')
Papa_face_encoding = face_recognition.face_encodings(Papa_image)[0]

Mama_image = face_recognition.load_image_file('photo_2020-07-10_02-18-52.jpg')
Mama_face_encoding = face_recognition.face_encodings(Mama_image)[0]


known_face_encodings = [
    Alexandra_face_encoding,
    Sergey_face_encoding,
    Evgeniy_face_encoding,
    Papa_face_encoding,
    Mama_face_encoding]

known_face_names = [
    'Alexandra_Preobrazhenskaya',
    'SergeySergeevich',
    'Evgeniy_Alexandrovich',
    'Alexandr_Nikolaevich',
    'Alexandra_Sergeevna'
]


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
face_position = []
process_this_frame = True
face_landmarks = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    #Postion Frame
    Direction_frame = cv2.resize(frame, (50, 50), fx=1.50, fy=1.50)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_position = face_recognition.face_landmarks(rgb_small_frame, face_locations)
        face_position = face_recognition.face_landmarks(rgb_small_frame, face_locations)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_landmarks = face_recognition.face_landmarks(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    def rect_to_bb(rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        # return a tuple of (x, y, w, h)
        return x, y, w, h

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 10, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Movement of a person
        if right < 448:
            Right_Command = "You are in the right side"
            cv2.putText(frame, Right_Command, (left - 100, bottom - 300), font, 1.0, (255, 255, 255), 1)

        if left > 928:
            Left_Command = "You are in the left side"
            cv2.putText(frame,  Left_Command, (left - 100, bottom - 300), font, 1.0, (255, 255, 255), 1)
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

video_capture.release()
cv2.destroyAllWindows()





