import csv
import cv2
import numpy as np

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


if __name__ == '__main__':
	X_train, y_train = load_data()