import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import pickle
import GUI_Tkinker as GUItk

base_path = GUItk.run_input_box("Input the path to directory containing your dataset folder: ")
dataset_folder_name = GUItk.run_input_box("Input the name of your dataset folder: ")

dataset_dir = os.path.join(base_path, dataset_folder_name)

IMG_SIZE = int(GUItk.run_input_box("Input the target image size: "))

CATEGORIES = []

for category_name in os.listdir(dataset_dir):
     CATEGORIES.append(category_name)

def Create_images_array():
    for category in CATEGORIES:
        img_path = os.path.join(dataset_dir, category)
        for img in os.listdir(img_path):
            img_array = cv2.imread(os.path.join(img_path,img), cv2.IMREAD_GRAYSCALE) #convert images into a grayscale array
            #print(img_array)
            #print(img_array.shape) #prints the original size of an image example
            #plt.imshow(img_array, cmap="gray") #shows the example of an image before resizing
            #plt.show()
            
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #resize images to 50x50 pixels
            #print(new_array)
            #print(new_array.shape) #prints the size of the image example after resizing to 50x50 pixels
            #plt.imshow(new_array, cmap="gray") #shows the example of an image after resizing to 50x50 pixels
            #plt.show()
            
            break
        break

def Create_training_data():

    training_data = []
    
    for category in CATEGORIES:
        path = os.path.join(dataset_dir, category)
        class_num = CATEGORIES.index(category) #each category prescribed an index number
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #convert images into a grayscale array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    random.shuffle(training_data)
    return training_data

def Pack_data(training_data):
    x_train = [] #features set
    y_train = [] #labels

    for features, label in training_data:
        x_train.append(features)
        y_train.append(label)

    x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # -1 = how many features, 1 = grayscale, 3 = RGB
    y_train = np.array(y_train)

    # saving

    pickle_out = open("X.pickle", "wb")
    pickle.dump(x_train, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

def Load_data():
    x_train = pickle.load(open("X.pickle", "rb"))
    y_train = pickle.load(open("y.pickle", "rb"))
    return x_train, y_train

def Normalize_data(x_train):
    x_train = x_train/255.0
    return x_train