import sys
import os
import dlib
import glob
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score

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
        



        
    
def recognize_face(face, embeddings, names):
    
    
    
    euclidean_distance_array = []
    # Loop over  names and encodings.
    for i, embedding in enumerate(embeddings):
        euclidean_distance = np.linalg.norm(np.array(face)-np.array(embedding))
        euclidean_distance_array.append(euclidean_distance)

    distances = np.array(euclidean_distance_array)
    min = distances.argmin()
    if distances[min] <0.6: 
        name = names[min]
        return name

        

cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3
while True:
    ret_val, img = cam.read()
    #img = cv2.resize(img, dsize = None, fx = 0.5, fy=0.5)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image,1)
    #live_embedding = []
    for i, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = sp(rgb_image, d)
        live_embedding = facerec.compute_face_descriptor(rgb_image,shape)  
        g = clf.predict([live_embedding])
        #print(g[0])
        #cv2.rectangle(img,(d.left(), d.top()), (d.right(), d.bottom()), color_green, line_width)
        proba = clf.predict_proba([live_embedding])
        #rint(proba)
        for index, prob in enumerate(proba[0]):
            if (prob>0.46):
                print(prob)
                #cv2.putText(img, g[0], (d.left(), d.top()),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
      
