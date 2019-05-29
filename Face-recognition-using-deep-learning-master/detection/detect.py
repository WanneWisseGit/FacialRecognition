import imutils
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import pickle
import time
import cv2
import os

dirname = os.path.dirname(__file__)
# pathToFaceDetector = os.path.join('face_detector/')
# dataset = os.path.join(pathToFaceDetector, 'deploy.prototxt')
# "res10_300x300_ssd_iter_140000.caffemodel"
pathToProto = os.path.join(dirname, 'caffeDetector/deploy.prototxt')
pathToDetector = os.path.join(dirname, 'caffeDetector/res10_300x300_ssd_iter_140000.caffemodel')
modelPath = os.path.join(dirname, 'net.model')
model = load_model(modelPath)
encoder = os.path.join(dirname, 'encoder.pickle')
readEncoder= pickle.loads(open(encoder, "rb").read())
# print(pathToProto)
net = cv2.dnn.readNetFromCaffe(pathToProto, pathToDetector)

# def adjust_gamma(image, gamma):
# 	invGamma = 1.0 / gamma
# 	table = np.array([((i / 255.0) ** invGamma) * 255
# 		for i in np.arange(0, 256)]).astype("uint8")
# 	return cv2.LUT(image, table)

vs = VideoStream(src=0).start()
time.sleep(2.0)
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	(h, w) = frame.shape[:2]
	x = cv2.resize(frame, (300, 300))

	# adjusted = adjust_gamma(x, 2.5)

	blob = cv2.dnn.blobFromImage(x, 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (32, 32))
			# face = adjust_gamma(face,2.5)
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = readEncoder.classes_[j]

			# draw the label and bounding box on the frame
			label = "{}: {:.4f}".format(label, preds[j])
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()


