import numpy as np
import argparse
import cv2
import os

dirname = os.path.dirname(__file__)
pathToProto = os.path.join(dirname, 'caffeDetector/deploy.prototxt')
pathToDetector = os.path.join(dirname, 'caffeDetector/res10_300x300_ssd_iter_140000.caffemodel')

inputVideoFake = cv2.VideoCapture(os.path.join(dirname, 'videos/fake.mp4'))
outputVideoFake = os.path.join(dirname, 'dataset/fake')
net = cv2.dnn.readNetFromCaffe(pathToProto, pathToDetector)
read = 0
saved = 0

# def adjust_gamma(image, gamma):
# 	invGamma = 1.0 / gamma
# 	table = np.array([((i / 255.0) ** invGamma) * 255
# 		for i in np.arange(0, 256)]).astype("uint8")
# 	return cv2.LUT(image, table)

while True:
	(grabbed, frame) = inputVideoFake.read()
	if not grabbed:
		break
	read += 1
	if read % 16 != 0:
		continue
	(h, w) = frame.shape[:2]
	x = cv2.resize(frame, (300, 300))

	# adjusted = adjust_gamma(x,2.5)

	blob = cv2.dnn.blobFromImage(x, 1.0, (300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()

	if len(detections) > 0 :
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			p = os.path.sep.join([outputVideoFake, "{}.png".format(saved)])
			cv2.imwrite(p, face)
			saved += 1
			print("[INFO] saved {} to disk".format(p))
inputVideoFake.release()
cv2.destroyAllWindows()