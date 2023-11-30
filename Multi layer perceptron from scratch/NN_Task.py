import Preprocess
import BackPropagation
import tkinter as tk
from tkinter import ttk
import numpy as np


def enable_train_button():
    # Enable the "Train Neural Network" button if all values are provided
    if all(entry.get() for entry in entry_widgets):
        train_button["state"] = "normal"
    else:
        train_button["state"] = "disabled"


def printMatrix(matrix):
  label = tk.Label(window, text=matrix[0][0], font=('bold', 12))
  label.place(relx=0.8, rely=0.71)
  label = tk.Label(window, text=matrix[0][1], font=('bold', 12))
  label.place(relx=0.88, rely=0.71)

  label = tk.Label(window, text=matrix[1][0], font=('bold', 12))
  label.place(relx=0.8, rely=0.79)
  label = tk.Label(window, text=matrix[1][1], font=('bold', 12))
  label.place(relx=0.88, rely=0.79)
  matrix2()


# Accuracy && Confusion Matrix
def matrix2():
    label = tk.Label(window, text='Accuracy', font=('bold', 17))
    label.place(relx=0.78, rely=0.4)
    label = tk.Label(window, text='Confusion Matrix', font=('bold', 17))
    label.place(relx=0.74, rely=0.6)
    label = tk.Label(window, text='Predicted Class', font=('bold', 12))
    label.place(relx=0.78, rely=0.65)
    label = tk.Label(window, text='Actual Class', font=('bold', 12))
    label.place(relx=0.65, rely=0.75)

    label = tk.Label(window, text='-1', font=('bold', 12))
    label.place(relx=0.8, rely=0.68)
    label = tk.Label(window, text='1', font=('bold', 12))
    label.place(relx=0.88, rely=0.68)

    label = tk.Label(window, text='-1', font=('bold', 12))
    label.place(relx=0.75, rely=0.71)
    label = tk.Label(window, text='1', font=('bold', 12))
    label.place(relx=0.75, rely=0.79)


def display_confusion_matrix(bp, y_test, y_pred, title):
    cm = bp.confusion_matrix(y_pred, y_test)

    # Create a new Tkinter window
    window = tk.Tk()
    window.title(f"Confusion Matrix of {title}")
    classes = ['BOMBAY', 'CALI', 'SIRA']

    style = ttk.Style()
    style.configure("TLabel", padding=5, font=('bold', 10))

    for i, class_name in enumerate(classes):
        label_col = ttk.Label(window, text=class_name, style="TLabel")
        label_col.grid(row=0, column=i + 1, sticky="nsew")

    for i, class_name in enumerate(classes):
        label_row = ttk.Label(window, text=class_name, style="TLabel")
        label_row.grid(row=i + 1, column=0, sticky="nsew")

    for i in range(len(classes)):
        for j in range(len(classes)):
            label = ttk.Label(window, text=f"{cm[i][j]}", style="TLabel")
            label.grid(row=i + 1, column=j + 1, sticky="nsew")

    acc = bp.classification_accuracy(cm)
    accuracy_label = ttk.Label(window, text=f"Accuracy: {acc:.2%}", style="TLabel")
    accuracy_label.grid(row=len(classes) + 1, column=0, columnspan=len(classes) + 1)

    # Configure row and column weights
    for i in range(len(classes) + 2):
        window.grid_rowconfigure(i, weight=1)
        window.grid_columnconfigure(i, weight=1)

    # Run the Tkinter event loop
    window.mainloop()


def buttonPressed():
    hidden_layers = int(hidden_layers_entry.get())
    learning_rate = float(learning_rate_entry.get())
    epochs = int(epochs_entry.get())
    add_bias = bias_var.get()
    activation_function = activation_function_var.get()

    # Close the current window
    window.destroy()

    input_values = []  # To store input values for each hidden layer

    for layer in range(hidden_layers):
        # Open a new window that takes one input only
        new_window = tk.Tk()
        new_window.title(f"New Window - Layer {layer + 1}")

        # Create and place the entry widget in the new window
        new_entry_label = ttk.Label(new_window, text=f"Enter input for Layer {layer + 1}:")
        new_entry_label.grid(row=0, column=0, padx=10, pady=10)
        new_entry = ttk.Entry(new_window)
        new_entry.grid(row=0, column=1, padx=10, pady=10)

        def process_input_layer():
            # Access the input from the new entry widget
            input_value = int(new_entry.get())
            input_values.append(input_value)
            # Close the current window
            new_window.destroy()

        # Create and place the button to process the input in the new window
        process_button = ttk.Button(new_window, text="Process Input", command=process_input_layer)
        process_button.grid(row=1, column=0, columnspan=2, pady=10)

        new_window.mainloop()
        # Training the neural network with the generated data
    prep = Preprocess.PreProcessing()
    bp = BackPropagation.BackPropagation(num_of_hidden_layers=hidden_layers, hidden_layer_neurons=input_values,
                                         learning_rate=learning_rate, num_of_iterations=epochs, bias=add_bias,
                                         activation_function=activation_function)
    weight_input, weights_hidden, weight_output, y_pred_arr = bp.fit(np.array(prep.X_train1), np.array(prep.y_train1))
    y_train = np.argmax(np.array(prep.y_train1), axis=1)
    display_confusion_matrix(bp, y_train, y_pred_arr, "Train")
    y_pred = bp.predict(np.array(prep.X_test1), np.array(prep.y_test1), weight_input, weights_hidden, weight_output)
    y_test = np.argmax(np.array(prep.y_test1), axis=1)
    display_confusion_matrix(bp, y_test, y_pred, "Test")


window = tk.Tk()
window.title("Neural Network Configuration")

# Get screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Set window width and height
window_width = 320
window_height = 130

# Calculate x and y coordinates for the center of the screen
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"{window_width}x{window_height}+{x}+{y}")
hidden_layers_label = ttk.Label(window, text="Enter number of hidden layers:")
hidden_layers_label.grid(row=0, column=0, sticky=tk.W)
hidden_layers_entry = ttk.Entry(window)
hidden_layers_entry.grid(row=0, column=1)

# neurons_label = ttk.Label(window, text="Enter number of neurons in each hidden layer:")
# neurons_label.grid(row=1, column=0, sticky=tk.W)
# neurons_entry = ttk.Entry(window)
# neurons_entry.grid(row=1, column=1)

learning_rate_label = ttk.Label(window, text="Enter learning rate (eta):")
learning_rate_label.grid(row=2, column=0, sticky=tk.W)
learning_rate_entry = ttk.Entry(window)
learning_rate_entry.grid(row=2, column=1)

epochs_label = ttk.Label(window, text="Enter number of epochs (m):")
epochs_label.grid(row=3, column=0, sticky=tk.W)
epochs_entry = ttk.Entry(window)
epochs_entry.grid(row=3, column=1)

bias_var = tk.BooleanVar()
bias_checkbox = ttk.Checkbutton(window, text="Add bias", variable=bias_var)
bias_checkbox.grid(row=4, column=0, columnspan=2, sticky=tk.W)

activation_function_label = ttk.Label(window, text="Choose activation function:")
activation_function_label.grid(row=5, column=0, sticky=tk.W)
activation_function_var = tk.StringVar()
activation_function_combobox = ttk.Combobox(window, textvariable=activation_function_var,values=["Sigmoid", "Hyperbolic Tangent"])
activation_function_combobox.grid(row=5, column=1)

train_button = ttk.Button(window, text="Train Neural Network", command=buttonPressed, state="disabled")
train_button.grid(row=6, column=0, columnspan=2)

entry_widgets = [hidden_layers_entry, learning_rate_entry, epochs_entry]

for entry in entry_widgets:
    entry.bind("<FocusIn>", lambda event, entry=entry: enable_train_button())
    entry.bind("<FocusOut>", lambda event, entry=entry: enable_train_button())


window.mainloop()


