# from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import pickle
import cv2
import os



dirname = os.path.dirname(__file__)
dataset = os.path.join(dirname, 'dataset')
netModel = os.path.join(dirname, 'net.model')
encoder = os.path.join(dirname, 'encoder.pickle')

INIT_LR = 1e-4
BS = 8
EPOCHS = 50
data = []
labels = []
# def adjust_gamma(image, gamma=1.0):
# 	invGamma = 1.0 / gamma
# 	table = np.array([((i / 255.0) ** invGamma) * 255
# 		for i in np.arange(0, 256)]).astype("uint8")
# 	return cv2.LUT(image, table)

for imagePath in list(paths.list_images(dataset)):
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))

	# adjusted = adjust_gamma(image, 2.5)

	data.append(image)
	labels.append(label)

data = np.array(data, dtype="float") / 255.0

#the net te detect fake en real faces
class realFakeNet:
	@staticmethod
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
		model.add(Conv2D(16, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		return model

#Encode fro the 2 labels real and fake
lableEncoder = LabelEncoder()
labels = lableEncoder.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = realFakeNet.build(width=32, height=32, depth=3,
	classes=len(lableEncoder.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# training network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

predictions = model.predict(testX, batch_size=BS)
model.save(netModel)

x = open(encoder, "wb")
x.write(pickle.dumps(lableEncoder))
x.close()
