import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#import preprocessing_data as prep
import GUI_Tkinker as GUItk
import plotting_learning_curves as plot


test_dataset_dir = GUItk.run_input_box("Input the path to the test images folder: ")

CATEGORIES = ['Alternating current source', 'Ammeter', 'Battery', 'Capacitor', 
               'DC Voltage source (type 1)', 'DC Voltage source (type 2)', 
              'Dependent current source', 'Dependent voltage source', 'Diode', 'Direct current source',
              'GND (type 1)', 'GND (type 2)', 'Inductor', 'Resistor', 'Voltmeter']


CATEGORIES1 = []

dataset_dir = 'C:/BADASSIUM/IB/CS/INTERNAL/TESTTTTTTs/SolvaDataset_200_v3'

for folder_name in os.listdir(dataset_dir):
     CATEGORIES1.append(folder_name)

print(CATEGORIES1)


CATEGORIES2 = []

dataset_dir = 'C:/BADASSIUM/IB/CS/INTERNAL/datasets/SolvaDataset_200_v3'

for folder_name in os.listdir(dataset_dir):
     CATEGORIES2.append(folder_name)

print(CATEGORIES2)
'''
def prepare(use_img_path):
    IMG_SIZE = 50
    img_array = cv2.imread(use_img_path, cv2.IMREAD_GRAYSCALE) 
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array / 255.0
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("models/128-neurons-0-dense-4-conv-(3, 3)-kernel-1696156039.model")

prediction_p = model.predict(prepare("test3.bmp"))
print(CATEGORIES[np.argmax(prediction_p[0])])

# Display the prediction values
df = pd.DataFrame(prediction_p[0], columns=['Probability'])
df['Component'] = CATEGORIES
df = df[['Component', 'Probability']]
print(df)
'''

model = tf.keras.models.load_model("C:/BADASSIUM/IB/CS/INTERNAL/models/128-neurons-0-dense-4-conv-(3, 3)-kernel-1696156039.model")
model2 = plot.best_model_name
#model = tf.keras.models.load_model("C:/BADASSIUM/IB/CS/INTERNAL/TESTTTTTTs/models/256-neurons-0-dense-4-conv-(3, 3)-kernel.model")



'''
image = tf.keras.preprocessing.image.load_img("test4.bmp", color_mode='grayscale', target_size=(50, 50))
image_array = tf.keras.preprocessing.image.img_to_array(image)

# Normalize the image
image_array = image_array / 255.0

# Predict the digit
predictions = model.predict(np.expand_dims(image_array, axis=0))[0]
predicted_digit = np.argmax(predictions)

# Display the prediction values
df = pd.DataFrame(predictions, columns=['Probability'])
df['Comp'] = CATEGORIES
df = df[['Comp', 'Probability']]
print(df)

print(predicted_digit)
'''


for image_name in os.listdir(test_dataset_dir):
    image_path = os.path.join(test_dataset_dir, image_name)
    image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(50, 50))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # Normalize the image
    image_array = image_array / 255.0

    # Invert the colors
    image_array = 1.0 - image_array
    
    # Predict the digit
    predictions = model.predict(np.expand_dims(image_array, axis=0))[0]
    predicted_digit = np.argmax(predictions)
    predicted_CAT = CATEGORIES[predicted_digit]


    # Display the prediction values
    df = pd.DataFrame(predictions, columns=['Probability'])
    df['Digit'] = CATEGORIES
    df = df[['Digit', 'Probability']]
    print(df)

    print(image_name)
    print(predicted_CAT)

    # Display the image
    plt.imshow(image_array.reshape(50, 50), cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.axis('off')
    plt.show()

    GUItk.run_output_box(f"{image_name} \n {df} \n Predicted component: {predicted_CAT}")





# choosing best model
# diplaying the answer
# opening tensorflow
