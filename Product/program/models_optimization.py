'''
Change:
optimizer adam
within optimizer: learning rate
dense layers number 0:
how many units per layer/nodes per layer/layer size: eg Conv2d 64
activation units
kernel size (3, 3)
stride

here:
nuber of layers
nodes per layer
dense layers number 0:
'''

import time
import pickle
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
#from tensorflow.keras.callbacks import TensorBoard
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.

# for using only a specific part of GPU to run multiple models simpulaneously

#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.33)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

os.chdir("C:/BADASSIUM/IB/CS/INTERNAL/program")

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

#X = np.reshape(X, (X.shape[0], *X.shape[1:], 1)) --------------------------------

X = X/255.0 # Normalization

# the initial value in between
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            # Layer 1

            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Conv2D(layer_size, (3,3), input_shape = X.shape[1:])) # 3x3 pixels window, shape dynamically defined of X, here img_size = 50x50
            # After Conv2D, activation or pooling
            model.add(tf.keras.layers.Activation("relu"))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2))) # 2x2 pooled pixel window


            for layer in range(conv_layer-1):
                # Layer 2
                # Again

                model.add(tf.keras.layers.Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
                model.add(tf.keras.layers.Activation("relu"))
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
                # we have a 2x64 layer CNN

            # Layer 3

            model.add(tf.keras.layers.Flatten())

            for layer in range(dense_layer):
                model.add(tf.keras.layers.Dense(layer_size))
                model.add(tf.keras.layers.Activation("relu"))

            # output layer

            model.add(tf.keras.layers.Dense(1))
            model.add(tf.keras.layers.Activation('sigmoid')) # or model.add(Dense(1), activation='sigmoid)

            # this will save the model's training data to logs/NAME, which can then be read by TensorBoard:
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss="binary_crossentropy",
                        optimizer="adam",
                        metrics=['accuracy'])

            history = model.fit(X, y, 
                    batch_size=32, 
                    epochs=3, 
                    validation_split=0.1, 
                    callbacks=[tensorboard]) # how many to pass at one time, validation set = 10% of all
            
            # Plot training loss vs validation loss
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            #plt.show()

            try:
                #filename = f"figure_smooth_case_{case}_value_{value}.png"
                plt.savefig(f"C:/BADASSIUM/IB/CS/INTERNAL/plots/{NAME}.png")
                #print(f"Figure_smooth_case_{case}_value_{value}_saved")
                plt.close() # close the current figure
            except Exception as e:
                print(e)


            model.save('electrical_components.model')

#tensorboard --logdir logs/
'''
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))# Dense layer with 128 neurons and ReLU function activation
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))# Output layer with 10 neurons (for each digit) and softmax activation

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss="binary_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])

history = model.fit(X, y, 
        batch_size=32, 
        epochs=3, 
        validation_split=0.1, 
        callbacks=[tensorboard]) # how many to pass at one time, validation set = 10% of all

# Plot training loss vs validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
#plt.show()
'''

'''
def load_data():
    # Load labels from CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    labels_df = pd.read_csv('training/labels.csv')
    images = []
    labels = []
    for index, row in labels_df.iterrows():
        image_path = os.path.join('training/images', row['Image Name'])
        image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale')
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        images.append(image_array)
        labels.append(int(row['Label']))
    return np.array(images), np.array(labels)

    images, labels = load_data()
    images = preprocess_images(images)
    labels = tf.keras.utils.to_categorical(labels)
'''