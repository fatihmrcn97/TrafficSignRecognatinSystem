
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

import cv2
import os
import sys
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow.keras as tfk
import datetime

save_path = os.path.join('BestModel')

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.1

class Build_Model():
    
    
    def main():
        
    	a = Build_Model()
    	# Check command-line arguments
    	if len(sys.argv) not in [1, 3]:
    		sys.exit("Usage: python traffic.py data_directory [model.h5]")
    
    	# Get image arrays and labels for all image files
    	images, labels = a.load_data(os.path.dirname(sys.argv[0]))
    
    	# Split data into training and testing sets
    	labels = tfk.utils.to_categorical(labels)
        # First we divide our data set train 90 , test 10 and 
    	x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=TEST_SIZE)
        # Validation set we have train 90 and we divide 22.2 and we can approximatly 70 train , 20 validation
    	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.22)    
    	# Get a compiled neural network
    	model = a.get_model()
    	callback = tfk.callbacks.EarlyStopping(monitor='loss', patience=3)
        #TensorBoard Usage
    	logdir = "logs\\model\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    	tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch = 100000000)
    	model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=["accuracy"])
    	model.summary()
    	checkpoint_filepath = 'weights.{epoch:02d}'
    	complete_name = os.path.join(save_path, checkpoint_filepath)
    	model_checkpoint_callback = tfk.callbacks.ModelCheckpoint(
            filepath=complete_name,
            save_weights_only=True,
            monitor='accuracy',
            mode='max',
            save_best_only=True)
    	# Fit model on training data
    	history =model.fit(x_train, y_train, epochs=EPOCHS,callbacks=[callback,model_checkpoint_callback,tensorboard_callback],validation_data=(x_val,y_val))
    	print(len(history.history['loss'])," Only X epochs are run.")
    	# Evaluate neural network performance
    	model.evaluate(x_test, y_test, verbose=2)
    	print(len(sys.argv))
    	# Save model to file
    	if len(sys.argv) == 1:
    		filename = os.path.dirname(sys.argv[0])
    		model.save("saved_model.h5")
    		print(f"Model saved to {filename}.")
    
    def load_data(self,data_dir):
    	data = []
    	labels = []
    	for i in range(NUM_CATEGORIES):
    		path = os.path.join(data_dir, "gtsrb", str(i))
    		images = os.listdir(path)
    		for j in images:
    			try:
    				image = cv2.imread(os.path.join(path, j))
    				image_from_array = Image.fromarray(image, 'RGB')
    				resized_image = image_from_array.resize((IMG_HEIGHT, IMG_WIDTH))
    				data.append(np.array(resized_image))
    				labels.append(i)
    			except AttributeError:
    				print("Error loading the image!")

    	images_data = (data, labels)
    	return images_data

    def get_model(self):
    	# initialize the model along with the input shape to be
    	# "channels last" and the channels dimension itself
    	model = Sequential()
    	inputShape = (IMG_WIDTH, IMG_HEIGHT, 3)
    	chanDim = -1
    	# CONV => RELU => BN => POOL
    	model.add(Conv2D(8, (5, 5), padding="same",input_shape=inputShape))
    	model.add(Activation("relu"))
    	model.add(BatchNormalization(axis=chanDim))
    	model.add(MaxPooling2D(pool_size=(2, 2)))
    	# first set of (CONV => RELU => CONV => RELU) * 2 => POOL
    	model.add(Conv2D(16, (3, 3), padding="same"))
    	model.add(Activation("relu"))
    	model.add(BatchNormalization(axis=chanDim))
    	model.add(Conv2D(16, (3, 3), padding="same"))
    	model.add(Activation("relu"))
    	model.add(BatchNormalization(axis=chanDim))
    	model.add(MaxPooling2D(pool_size=(2, 2)))
    	# second set of (CONV => RELU => CONV => RELU) * 2 => POOL
        
    	# model.add(Conv2D(32, (3, 3), padding="same"))
    	# model.add(Activation("relu"))
    	# model.add(BatchNormalization(axis=chanDim))
    	# model.add(Conv2D(32, (3, 3), padding="same"))
    	# model.add(Activation("relu"))
    	# model.add(BatchNormalization(axis=chanDim))
    	# model.add(MaxPooling2D(pool_size=(2, 2)))
        
    	# first set of FC => RELU layers
    	model.add(Flatten())
    	model.add(Dense(128))
    	model.add(Activation("relu"))
    	model.add(BatchNormalization())
    	model.add(Dropout(0.5))
    	# second set of FC => RELU layers
    	model.add(Flatten())
    	model.add(Dense(128))
    	model.add(Activation("relu"))
    	model.add(BatchNormalization())
    	model.add(Dropout(0.5))
    	# softmax classifier
    	model.add(Dense(NUM_CATEGORIES))
    	model.add(Activation("softmax"))
    	return model



if __name__ == "__main__":
    Build_Model.main()