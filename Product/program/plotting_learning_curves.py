import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import models_training_and_optimization as train

def load_saved_histories_and_colors(histories, colors):
    with open(os.path.join(train.histories_dir, "histories.pkl"), 'rb') as f:
        histories = pickle.load(f)
    
    with open(os.path.join(train.colors_dir, "colors.pkl"), 'rb') as f:
        colors = pickle.load(f)
    print("Loaded Histories:", histories)

    return histories, colors

def plot_learning_curves(histories, colors):
    print("Loaded Histories:", histories)
    metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']

    for metric in metrics:

        plt.figure(figsize=(20,30))
        plt.grid(True, color='lightgrey')
        
        
        reverse_order = True if 'loss' or 'val_loss' == metric else False #Sort the histories based on their final metric value
        sorted_names = sorted(histories.keys(), key=lambda x: histories[x][metric][-1], reverse=reverse_order)

        for NAME in sorted_names:
            color = colors[NAME]
            plt.plot(histories[NAME][metric], label=NAME, color=color)
        

        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.grid(color='lightgrey')
        plt.title(f'{metric.capitalize()} Over Epochs')

        plt.savefig(os.path.join(train.plots_dir, f"{metric}_Graph.png"))
        plt.close()

        
        legend_labels = [mpatches.Patch(color=colors[NAME], label=f"{NAME} ({histories[NAME][metric][-1]:.4f})") for NAME in sorted_names] #Create a legend figure
        fig_leg = plt.figure(figsize=(5, 5))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.legend(handles=legend_labels)
        ax_leg.axis('off')

        fig_leg.savefig(os.path.join(train.plots_dir, f"{metric}_Legend.png"), bbox_inches='tight')
        plt.close()

        
        sorted_names_by_val_acc = sorted(histories.keys(), key=lambda x: histories[x]['val_accuracy'][-1], reverse=True)#Identify the best model with the highest final validation accuracy
        best_model_name = sorted_names_by_val_acc[0]
        
        
        #Best model's accuracy graph
        plt.figure(figsize=(10, 10))
        plt.plot(histories[best_model_name]['accuracy'], label='Training Accuracy', color="lightblue")
        plt.plot(histories[best_model_name]['val_accuracy'], label='Validation Accuracy', color="blue")
        plt.title(f"Accuracy curves for best model: {best_model_name}")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(train.plots_dir, f"Best_Model_Accuracy.png"))
        plt.close()

        #Best model's loss graph
        plt.figure(figsize=(10, 10))
        plt.plot(histories[best_model_name]['loss'], label='Training Loss', color="lightblue")
        plt.plot(histories[best_model_name]['val_loss'], label='Validation Loss', color="blue")
        plt.title(f"Loss curves for best model: {best_model_name}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(train.plots_dir, f"Best_Model_Loss.png"))
        plt.close()