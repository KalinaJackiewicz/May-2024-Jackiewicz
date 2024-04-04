import tkinter as tk

def run_input_box(prompt):
    def on_submit(event=None):
        
        user_input[0] = input_box.get()
        root.destroy() #Closes the input box after the user clicks the submit button or the Enter key on the keyboard

    root = tk.Tk()#Initializing the box
    root.title("Input Box")
    label = tk.Label(root, text=prompt)
    label.pack()

    input_box = tk.Entry(root, width=50)#Creates the box
    input_box.pack()
    input_box.focus() #Focus the input box

    root.bind("<Return>", on_submit)
    input_box.bind("<Return>", on_submit)
    submit_button = tk.Button(root, text="Enter", command=on_submit)#submit button
    submit_button.pack()

    user_input = [None]#a list stores the input
    root.mainloop()
        
    return user_input[0]


def run_input_box_with_parameters():
    def on_submit():

        try:

            results['dense_layers_nums'] = [int(x) for x in dense_layers_input.get().split(',')]
            results['neurons_per_layer'] = [int(x) for x in neurons_input.get().split(',')]
            results['conv_layers_nums'] = [int(x) for x in conv_layers_input.get().split(',')]
            
            input_string = kernel_sizes_input.get().replace(" ", "")
            tuple_strings = input_string[1:-1].split("),(") if input_string.startswith("(") else input_string.split("),(")
            results['kernel_sizes'] = [tuple(map(int, x.split(','))) for x in tuple_strings]

        finally:
            root.destroy()

    root = tk.Tk()
    root.title("Configuration Inputs")

    #Dense layers input
    tk.Label(root, text="Input the numbers of dense layers (lowest possible nuber: 0) in the format 0,1,2...").pack()
    dense_layers_input = tk.Entry(root)
    dense_layers_input.pack()

    #Neurons per layer input
    tk.Label(root, text="Input the numbers of neurons per layer (in powers of 2) in the format 32,64,128...").pack()
    neurons_input = tk.Entry(root)
    neurons_input.pack()

    #Convolutional layers input
    tk.Label(root, text="Input the numbers of convolutional layers (lowest possible nuber: 1) in the format 1,2,3...:").pack()
    conv_layers_input = tk.Entry(root)
    conv_layers_input.pack()

    #Kernel sizes input
    tk.Label(root, text="Input the kernel matrice sizes (lowest possible size: (1,1)) in the format (1,1),(2,2),...:").pack()
    kernel_sizes_input = tk.Entry(root)
    kernel_sizes_input.pack()

    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.pack()

    results = {}
    root.mainloop()

    return (results.get('dense_layers_nums', []),
            results.get('neurons_per_layer', []),
            results.get('conv_layers_nums', []),
            results.get('kernel_sizes', []))


def run_output_box(prompt):
    def on_submit():
        root.destroy()

    root = tk.Tk()
    root.title("Output Box")
    label = tk.Label(root, text=prompt)
    label.pack()
    submit_button = tk.Button(root, text="Enter", command=on_submit)
    submit_button.pack()

    root.mainloop()