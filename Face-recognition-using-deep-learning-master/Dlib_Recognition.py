

#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to use dlib's face recognition tool.  This tool maps
#   an image of a human face to a 128 dimensional vector space where images of
#   the same person are near to each other and images from different people are
#   far apart.  Therefore, you can perform face recognition by mapping faces to
#   the 128D space and then checking if their Euclidean distance is small
#   enough. 
#
#   When using a distance threshold of 0.6, the dlib model obtains an accuracy
#   of 99.38% on the standard LFW face recognition benchmark, which is
#   comparable to other state-of-the-art methods for face recognition as of
#   February 2017. This accuracy means that, when presented with a pair of face
#   images, the tool will correctly identify if the pair belongs to the same
#   person or is from different people 99.38% of the time.
#
#   Finally, for an in-depth discussion of how dlib's tool works you should
#   refer to the C++ example program dnn_face_recognition_ex.cpp and the
#   attendant documentation referenced therein.
#
#
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import sys
import os
import dlib
import glob
import numpy as np
import cv2

print(dlib.DLIB_USE_CUDA)
dlib.hit_enter_to_continue()

predictor_path = "shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "images/"
embeddings = []
names = []
# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)



# Now process all the images
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    name = os.path.splitext(os.path.basename(f))[0]

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
      
        
        # Let's generate the aligned image using get_face_chip
        face_chip = dlib.get_face_chip(img, shape)    
        # Now we simply pass this chip (aligned image) to the api
        face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(img,shape,1)                
        embeddings.append(face_descriptor_from_prealigned_image)
        names.append(name)      
        


testimg = dlib.load_rgb_image("jari.jpg")
dets = detector(testimg, 1)


        
    
def recognize_face(embedding, input_embeddings, model):

    
    
    minimum_distance = 500
    name = None
    dis = 0
    euclidean_distance_array = []
    # Loop over  names and encodings.
    for idx, input_embedding in enumerate(input_embeddings):
        euclidean_distance = np.linalg.norm(np.array(embedding)-np.array(input_embedding))
        euclidean_distance_array.append(euclidean_distance)

    distances = np.array(euclidean_distance_array)
    min = distances.argmin()
    if distances[min] <0.6: 
        name = names[min]
        return name

        
    

detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3
while True:
    ret_val, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image,1)
    live_embedding = []
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = sp(rgb_image, d)
        live_embedding = facerec.compute_face_descriptor(rgb_image,shape,1)  
        name = recognize_face(live_embedding,embeddings,names)
        print(name)
        cv2.rectangle(img,(d.left(), d.top()), (d.right(), d.bottom()), color_green, line_width)
        cv2.putText(img, name, (d.left(), d.top()),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
      
