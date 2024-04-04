import os
import random
import tensorflow as tf
import pickle
import preprocessing_data as prep
import GUI_Tkinker as GUItk

plots_dir = os.path.join(prep.base_path, "plots")
histories_dir = os.path.join(plots_dir, "histories")
colors_dir = os.path.join(plots_dir, "colors")


dense_layers_nums, neurons_per_layer, conv_layers_nums, kernel_sizes = GUItk.run_input_box_with_parameters()

number_of_models = len(dense_layers_nums) * len(neurons_per_layer) * len(conv_layers_nums) * len(kernel_sizes)
model_num = 1

def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def Set_NAME(neurons_num, dense_layers_num, conv_layers_num, kernel_size):
    NAME = "{}-neurons-{}-dense-{}-conv-{}-kernel".format(neurons_num, dense_layers_num, conv_layers_num, kernel_size)
    return NAME

class CNN_Model:

    def __init__(self, x_train, CATEGORIES_num, neurons_num, dense_layers_num, conv_layers_num, kernel_size):
        self.model = self.Define_model_architecture(x_train, CATEGORIES_num, neurons_num, dense_layers_num, conv_layers_num, kernel_size)

    def Define_model_architecture(self, x_train, CATEGORIES_num, neurons_num, dense_layers_num, conv_layers_num, kernel_size):

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(neurons_num, kernel_size, input_shape = x_train.shape[1:])) # 3x3 pixels window, shape dynamically defined of X, here img_size = 50x50
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        for _ in range(conv_layers_num-1):

            model.add(tf.keras.layers.Conv2D(neurons_num, kernel_size, input_shape = x_train.shape[1:])) # 3x3 pixels window, shape dynamically defined of X, here img_size = 50x50
            model.add(tf.keras.layers.Activation("relu"))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        
        model.add(tf.keras.layers.Flatten())

        for _ in range(dense_layers_num):
            model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.Activation("relu"))

        model.add(tf.keras.layers.Dense(len(prep.CATEGORIES))) #for each category
        model.add(tf.keras.layers.Activation("softmax"))

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy']) # what to track
        return model
        
    def Train_model(self, x_train, y_train, tensorboard):
        history = self.model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.3, callbacks=[tensorboard])
        return history
    
    def Save_model(self, NAME):
        self.model.save('models/{}.model'.format(NAME))

def random_color():
    return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def save_histories_and_colors(histories, colors):
    check_and_create_dir(histories_dir)
    with open("{}/histories.pkl".format(histories_dir), "wb") as f:
        pickle.dump(histories, f)

    check_and_create_dir(colors_dir)  
    with open("{}/colors.pkl".format(colors_dir), "wb") as f:
        pickle.dump(colors, f)

def Save_TensorBoard_logs(NAME):
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(NAME))
    return tensorboard