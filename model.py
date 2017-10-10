import csv
import cv2
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Lambda, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('epoch', 10, "Number of epochs used for training")
flags.DEFINE_string('batch_size', 32, "Batch size used for training")

def load_data():
	# Data
	images = []
	# Target
	measurements = []
	with open("./data/driving_log.csv") as csvfile:
		content = csv.reader(csvfile)
		for line in content:
			# Change image path as the learning has been done elsewhere
			center_image_path = line[0]
			center_image_name = center_image_path.split("\\")[-1]
			current_path = "./data/IMG/" + center_image_name
			# Load the image
			image = cv2.imread(current_path)
			images.append(image)
			# Load the steering angle
			measurements.append(line[3])
			# Augment data by flipping image around y-axis
			#images.append(cv2.flip(image, 1))
			#measurements.append(-line[3])

	# Return numpy arrays
	return (np.array(images), np.array(measurements))

def grayscale(input):
	from keras.backend import tf as ktf
	return ktf.image.rgb_to_grayscale(input)

def resize_image(input, w=32, h=32):
	from keras.backend import tf as ktf
	return ktf.image.resize_images(input, [w,h], ktf.image.ResizeMethod.BICUBIC)

def normalize_image(input):
	return (input / 255.0) - 0.5

def build_lenet_model(data):
	model = Sequential()

	# --- Convert image into grayscale
	model.add(Lambda(grayscale, input_shape=data.shape[1:]))
	# --- Resize it to have a 32x32 shape
	model.add(Lambda(resize_image))
	# --- Normalize and mean center the data
	model.add(Lambda(normalize_image))

	# --- Layer 1 : Convolution + ReLu activation + maxpooling
	model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

	# --- Layer 2 : Convolution + ReLu activation + maxpooling
	model.add(Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

	# --- Flatten the weights
	model.add(Flatten())

	# --- Layer 3 : Fully-connected + ReLu activation
	model.add(Dense(120))
	model.add(Activation('relu'))

	# --- Layer 4 : Fully-connected + ReLu activation
	model.add(Dense(84))
	model.add(Activation('relu'))

	# --- Layer 5 : Fully-connected
	model.add(Dense(1))

	print(model.summary()) 

	model.compile(optimizer='adam', loss='mse')	

	return model

def main(_):
	# Import training data
	X_train, y_train = load_data()
	print(X_train.shape, y_train.shape)

	# Build the model LeNet-5
	model = build_lenet_model(X_train)
	# Train the model
	model.fit(X_train, y_train, batch_size=int(FLAGS.batch_size), epochs=int(FLAGS.epoch), validation_split=0.2, shuffle=True)
	# Save it
	model.save('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()