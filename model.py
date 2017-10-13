import csv
import cv2
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras import regularizers
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Lambda, Activation, Dropout, ELU
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('epochs', 10, "Number of epochs used for training")
flags.DEFINE_string('batch_size', 32, "Batch size used for training")

def load_data():
	# Data
	images = list()
	# Target
	steering_angles = list()
	with open("./data/driving_log.csv") as csvfile:
		content = csv.reader(csvfile)
		next(content, None) # Remove header from csvfile
		for line in content:
			steering_center = float(line[3])

			# Create adjusted steering measurements for the side camera images
			correction = 0.2
			steering_left = steering_center + correction
			steering_right = steering_center - correction

			# Change image paths as the learning has been done elsewhere
			path = "./data/IMG/"
			img_center = cv2.imread(path + line[0].split("\\")[-1])
			img_left = cv2.imread(path + line[1].split("\\")[-1])
			img_right = cv2.imread(path + line[2].split("\\")[-1])

			# Load images and steering angles
			images.extend([img_center, img_left, img_right])
			steering_angles.extend([steering_center, steering_left, steering_right])

			# Augment data by flipping image around y-axis
			aug_img_center = cv2.flip(img_center, 1)
			aug_img_left = cv2.flip(img_left, 1)
			aug_img_right = cv2.flip(img_right, 1)
			images.extend([aug_img_center, aug_img_left, aug_img_right])
			steering_angles.extend([-1.0*steering_center, -1.0*steering_left, -1.0*steering_right])

	# Return numpy arrays
	return (np.array(images), np.array(steering_angles))

def grayscale(input):
	from keras.backend import tf as ktf
	return ktf.image.rgb_to_grayscale(input)

def resize_image(input, h, w):
	from keras.backend import tf as ktf
	return ktf.image.resize_images(input, [h,w], ktf.image.ResizeMethod.BICUBIC)

def normalize_image(input):
	return (input / 255.0) - 0.5

def build_lenet_model(data):
	model = Sequential()

	# --- Crop image to save only the region of interest
	model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=data.shape[1:]))
	# --- Convert image into grayscale
	model.add(Lambda(grayscale))
	# --- Resize it to have a 32x32 shape
	model.add(Lambda(resize_image, arguments={'h': 32, 'w': 32}))
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
	model.add(Dense(120, activity_regularizer=regularizers.l2(0.01)))
	model.add(Activation('relu'))

	# --- Layer 4 : Fully-connected + ReLu activation
	model.add(Dense(84, activity_regularizer=regularizers.l2(0.01)))
	model.add(Activation('relu'))

	# --- Layer 5 : Fully-connected
	model.add(Dense(1))

	model.compile(optimizer='adam', loss='mse')

	return model

def build_nvidia_model(data):
	model = Sequential()

	# --- Crop image to save only the region of interest
	model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=data.shape[1:]))
	# --- Normalize and mean center the data
	model.add(Lambda(normalize_image))

	# --- Layer 1 : Convolution + ReLu activation + maxpooling
	model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(1,1), kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2,2)))

	# --- Layer 2 : Convolution + ReLu activation + maxpooling
	model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(1,1), kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2,2)))

	# --- Layer 3 : Convolution + ReLu activation + maxpooling
	model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(1,1), kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2,2)))

	# --- Layer 4 : Convolution + ReLu activation
	model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())

	# --- Layer 5 : Convolution + ReLu activation
	model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())

	# --- Flatten the weights
	model.add(Flatten())

	# --- Layer 6 : Fully-connected + ReLu activation
	model.add(Dense(100, kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())

	# --- Layer 7 : Fully-connected + ReLu activation
	model.add(Dense(50, kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())

	# --- Layer 8 : Fully-connected + ReLu activation
	model.add(Dense(10, kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())

	# --- Layer 9 : Fully-connected
	model.add(Dense(1))

	print(model.summary())

	model.compile(optimizer='adam', loss='mse')

	return model

def main(_):
	# Import training data
	X_train, y_train = load_data()

	# Visualize data
	print('Image shape: {}, Steering data shape: {}'.format(X_train.shape, y_train.shape))
	plt.hist(y_train, bins=50)
	plt.xlabel('Steering angle')
	plt.title('Data distribution')
	plt.legend()
	plt.tight_layout()
	plt.show()
	plt.savefig('distribution.png')

	# Build the model
	#model = build_lenet_model(X_train)
	model = build_nvidia_model(X_train)
	print(model.summary())

	# Train the model
	history_object = model.fit(X_train, y_train, batch_size=int(FLAGS.batch_size), epochs=int(FLAGS.epochs), validation_split=0.2, shuffle=True)

	# Plot the training and validation loss after each epoch
	plt.plot(history_object['loss'])
	plt.plot(history_object['val_loss'])
	plt.title('Model mean squared error loss')
	plt.ylabel('Mean squared error loss')
	plt.xlabel('Epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()
	plt.savefig('training_validation_loss.png')

	# Save it
	model.save('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
