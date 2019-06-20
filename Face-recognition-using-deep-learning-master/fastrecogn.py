import sys
import os
import dlib
import glob
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score
import socket

print(dlib.DLIB_USE_CUDA)
dlib.hit_enter_to_continue()

predictor_path = "shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "SVM_Folder/"
embeddings = []
names = []

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)



# Now process all the images
for f in os.listdir(faces_folder_path):
    print(f)
    for i in os.listdir(faces_folder_path +"/"+ f + "/"):
        print(i)
        print("Processing file: {}".format(i))
        img = dlib.load_rgb_image(faces_folder_path + "/" + f + "/"+ i)
        name = os.path.splitext(os.path.basename(i))[0]
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        # Now process each face we found.
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)  
            # Now we simply pass this chip (aligned image) to the api
            face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(img,shape,100)  
                        
            embeddings.append(face_descriptor_from_prealigned_image)
            names.append(f)     

clf = svm.SVC(kernel='linear', probability=True)
svc = clf.fit(embeddings,names) 
        

        
face_names=[]
face_locations=[]
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3
process_this_frame = True
while True:
    ret, frame = cam.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    

    if process_this_frame:
        dets = detector(rgb_small_frame,1)
        #live_embedding = []
        face_names=[]
        face_locations=[]
        for i, face in enumerate(dets):
            predict = False
            shape = sp(rgb_small_frame, face)
            live_embedding = facerec.compute_face_descriptor(rgb_small_frame,shape)  
            name = clf.predict([live_embedding])[0]
            proba = clf.predict_proba([live_embedding])
            loc = []
            top = face.top()
            bottom = face.bottom()
            left = face.left()
            right = face.right()
            loc.append(top)
            loc.append(right)
            loc.append(bottom)
            loc.append(left)
            
            for index, prob in enumerate(proba[0]):
                if (prob>0.43):
                    predict = True
                    print(prob)
            if predict == True:
                face_locations.append(loc)
                face_names.append(name)
                MESSAGE = name
                sock = socket.socket(socket.AF_INET, # Internet
                                    socket.SOCK_DGRAM) # UDP
                sock.sendto(MESSAGE.encode(), ("127.0.0.1", 5005))
            else:
                face_locations.append(loc)
                face_names.append("")
    process_this_frame = not process_this_frame

     # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 1
        right *= 1
        bottom *= 1
        left *= 1

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                
    cv2.imshow('my webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
      
