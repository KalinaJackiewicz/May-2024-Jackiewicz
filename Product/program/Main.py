import os
import time
import subprocess
import webbrowser
import preprocessing_data as prep
import models_training_and_optimization as train
import plotting_learning_curves as plot
import GUI_Tkinker as GUItk

def main():
    
    os.chdir(prep.base_path)

    # preprocess and pack the dataset to X.picle and y.pickle

    prep.Create_images_array()
    training_data = prep.Create_training_data()
    prep.Pack_data(training_data)
    
    x_train, y_train = prep.Load_data()
    x_train = prep.Normalize_data(x_train=x_train)

    # train the models for different parameters

    histories = {}
    colors = {}

    model_num = 1

    for dense_layers_num in train.dense_layers_nums:
        for neurons_num in train.neurons_per_layer:
            for conv_layers_num in train.conv_layers_nums:
                for kernel_size in train.kernel_sizes:

                    NAME = train.Set_NAME(neurons_num, dense_layers_num, conv_layers_num, kernel_size)

                    if os.path.exists("models/{}.model".format(NAME)) == False:
                        MyModel = train.CNN_Model(x_train, len(prep.CATEGORIES), neurons_num, dense_layers_num, conv_layers_num, kernel_size)
                        tensorboard = train.Save_TensorBoard_logs(NAME)
                        history = MyModel.Train_model(x_train, y_train, tensorboard)
                        MyModel.Save_model(NAME)
                        print("Saved model {}/{}".format(model_num, train.number_of_models))

                        # Store this model's history and color
                        histories[NAME] = history.history
                        colors[NAME] = train.random_color()

                        model_num = model_num + 1

                    else:
                        GUItk.run_output_box("An old model in this folder with this name already exists - it will not be trained again. If you want to train a new one, please delete the old models first.")
                        pass

    train.save_histories_and_colors(histories, colors)

    # plot learning curve graphs for the trained models
    
    train.check_and_create_dir(train.plots_dir)
    histories, colors = plot.load_saved_histories_and_colors(histories, colors)
    plot.plot_learning_curves(histories, colors)


    # Run the tensorboard command and open logs online
    
    tensorboard_command = 'tensorboard --logdir logs'
    subprocess.Popen(tensorboard_command, shell=True)
    time.sleep(60)
    #print('opening website')
    os.system('start chrome http://localhost:6006/')
    time.sleep(10)
    GUItk.run_output_box("Your models are trained and their performances ready to evaluate.")
    


if __name__ == "__main__":
    main()