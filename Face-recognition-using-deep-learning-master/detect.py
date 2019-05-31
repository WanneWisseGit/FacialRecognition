import dlib
import cv2
face_detector = dlib.get_frontal_face_detector()
#face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
def scale_faces(face_rects, down_scale=1.5):
    faces = []
    for face in face_rects:
        scaled_face = dlib.rectangle(int(face.left() * down_scale),
                                    int(face.top() * down_scale),
                                    int(face.right() * down_scale),
                                    int(face.bottom() * down_scale))
        faces.append(scaled_face)
    return faces
def detect_faces(image, down_scale=1.5):
    image_scaled = cv2.resize(image, None, fx=1.0/down_scale, fy=1.0/down_scale, 
                              interpolation=cv2.INTER_LINEAR)
    faces = face_detector(image_scaled, 0)
  #  faces = [face.rect for face in faces]
    faces = scale_faces(faces, down_scale)
    return faces