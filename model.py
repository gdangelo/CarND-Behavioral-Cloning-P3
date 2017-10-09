import csv
import cv2
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Input, Dense, Flatten

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

	# Return numpy arrays
	return (np.array(images), np.array(measurements))

def main(_):
	# Import training data
	X_train, y_train = load_data()
	print(X_train.shape, y_train.shape)

	# Build the model
	model = Sequential()

	model.add(Flatten(input_shape=X_train.shape[1:]))
	model.add(Dense(1))

	print(model.summary()) 

	model.compile(optimizer='adam', loss='mse')

	# Train the model
	model.fit(X_train, y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epoch, validation_split=0.2, shuffle=True)

	# Save it
	model.save('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()