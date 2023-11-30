import numpy as np


class BackPropagation:
    def __init__(self, num_of_hidden_layers, hidden_layer_neurons=[], learning_rate=0.0001, num_of_iterations=1000,
                 bias=0, activation_function="Sigmoid"):
        # Hyper parameters
        self.input_layer_neurons = 5
        self.output_layer_neurons = 3
        self.num_of_hidden_layers = num_of_hidden_layers
        self.hidden_layer_neurons = hidden_layer_neurons
        self.learning_rate = learning_rate
        self.num_of_iterations = num_of_iterations
        self.bias = bias
        self.bias_hidden = []
        self.activation_function = activation_function
        self.results = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - x**2

    def confusion_matrix(self, y_actual, y_pred, num_classes=3):
        # Initialize confusion matrix
        conf_matrix = [[0] * num_classes for _ in range(num_classes)]

        for i in range(len(y_pred)):
            conf_matrix[y_actual[i]][y_pred[i]] += 1
        return conf_matrix

    def classification_accuracy(self, confusion_matrix):
        correct_predictions = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
        total_predictions = sum(sum(row) for row in confusion_matrix)
        print(correct_predictions / total_predictions)
        return correct_predictions / total_predictions

    def initialize_weights(self):
        # Weights
        np.random.seed(10)
        weights_hidden = []
        # Input layer weights
        weight_input = np.random.randn(self.input_layer_neurons, self.hidden_layer_neurons[0])

        # print("weight_input",weight_input.shape)
        weights_hidden.append(weight_input)
        if self.bias == 1:
            self.bias_hidden.append(np.random.randn(1, self.hidden_layer_neurons[0]))
        # Hidden layer weights
        for idx in range(1, self.num_of_hidden_layers):
            #  ("weight _hidden", np.random.randn(self.hidden_layer_neurons[idx-1], self.hidden_layer_neurons[idx]).shape)
            weights_hidden.append(np.random.randn(self.hidden_layer_neurons[idx - 1], self.hidden_layer_neurons[idx]))
            if self.bias == 1:
                self.bias_hidden.append(np.random.randn(1, self.hidden_layer_neurons[idx]))
        # Output layer weights
        if self.bias == 1:
            self.bias_hidden.append(np.random.randn(1, self.output_layer_neurons))
        weight_output = np.random.randn(self.hidden_layer_neurons[-1], self.output_layer_neurons)
        # print("weight_output", weight_output.shape)
        weights_hidden.append(weight_output)
        return weight_input, weights_hidden, weight_output, self.bias_hidden

    def feed_forward_propagation(self, x_train, weight_input, weights_hidden, weight_output, bias_hidden,
                                 activation_function):
        # Input layer
        #float_list = [float(item) for item in x_train]

        if self.bias == 1:
            input_net = np.dot(x_train, weight_input) + bias_hidden[0]
        else:
            input_net = np.dot(x_train, weight_input)
        # print(input_net.shape)
        if activation_function == "Sigmoid":
            input_layer_output = self.sigmoid(input_net)
        else:
            input_layer_output = self.tanh(input_net)

        # Hidden layer
        hidden_net = []
        hidden_layer_output = []
        hidden_layer_output.append(input_layer_output)
        for idx in range(1, len(weights_hidden) - 1):
            # second hidden layer which will require input layer's output
            if idx == 1:
                # print(np.dot(input_layer_output, weights_hidden[idx ]).shape)
                hidden_net.append(np.dot(input_layer_output, weights_hidden[idx]))
            # prevent out of bounds exception
            else:
                # print(np.dot(hidden_net[-1], weights_hidden[idx]).shape)
                hidden_net.append(np.dot(hidden_net[-1], weights_hidden[idx]))

            if activation_function == "Sigmoid":
                if self.bias == 1:
                    hidden_layer_output.append(self.sigmoid(hidden_net[-1] + bias_hidden[idx]))
                else:
                    hidden_layer_output.append(self.sigmoid(hidden_net[-1]))
            else:
                if self.bias == 1:
                    hidden_layer_output.append(self.tanh(hidden_net[-1] + bias_hidden[idx]))
                else:
                    hidden_layer_output.append(self.tanh(hidden_net[-1]))

                    # Output layer
        # print("-", np.dot(hidden_layer_output[-1], weight_output).shape)
        output_net = np.dot(hidden_layer_output[-1], weight_output)
        if self.bias == 1:
            output_net += bias_hidden[-1]
        output_net = np.squeeze(output_net)
        # print(np.squeeze(output_net))
        if activation_function == "Sigmoid":
            y_pred = self.sigmoid(output_net)
        else:
            y_pred = self.tanh(output_net)

        return y_pred, hidden_layer_output

    def back_propagation(self, hidden_layer_output, y_pred, x_train, y_train, weight_input, weights_hidden, weight_output,bias_hidden, activation_function):

        N = y_train.size

        # Calculating error and derivative output of each layer
        d_weight_hidden = []
        d_bias_hidden = []
        # Output layer
        output_layer_error = y_train - y_pred
        # sigmoid
        # print(len(hidden_layer_output))
        # print(len(hidden_layer_output[0]))
        if activation_function == "Sigmoid":
            d_weight_output = output_layer_error * y_pred * (1 - y_pred)
        else:
            d_weight_output = output_layer_error * self.tanh_derivative(y_pred)
        d_weight_hidden.append(d_weight_output)
        if self.bias == 1:
            d_bias_hidden.append(d_weight_output)
        if activation_function == "Sigmoid":
            k = 0
            for idx in reversed(weights_hidden):
                d1 = np.dot(d_weight_hidden[-1], idx.T)
                if self.bias == 1:
                    d2 = np.dot(d_bias_hidden[-1], bias_hidden[-1 - k].T)
                if (-1 - k) * -1 > len(hidden_layer_output):
                    break
                d_weight_hidden.append(d1 * hidden_layer_output[-1 - k] * (1 - hidden_layer_output[-1 - k]))
                if self.bias == 1:
                    d_bias_hidden.append(d2 * hidden_layer_output[-1 - k] * (1 - hidden_layer_output[-1 - k]))
                k += 1
        else:
            k = 0
            for idx in reversed(weights_hidden):
                d1 = np.dot(d_weight_hidden[-1], idx.T)
                if self.bias == 1:
                    d2 = np.dot(d_bias_hidden[-1], bias_hidden[-1 - k].T)
                if (-1 - k) * -1 > len(hidden_layer_output):
                    break
                d_weight_hidden.append(d1 * self.tanh_derivative(hidden_layer_output[-1 - k]))
                if self.bias == 1:
                    d_bias_hidden.append(d2 * self.tanh_derivative(hidden_layer_output[-1 - k]))
                k += 1

        d_weight_hidden.reverse()
        if self.bias == 1:
            d_bias_hidden.reverse()
        # Updating weights
        j = 0
        for i in d_weight_hidden:
            weights_hidden[j] = weights_hidden[j] + i * self.learning_rate
            j += 1

        if self.bias == 1:
            j = 0
            for i in d_bias_hidden:
                bias_hidden[j] = bias_hidden[j] + i * self.learning_rate
                j += 1
        weight_input = weights_hidden[0]
        weight_output = weights_hidden[-1]
        return weight_input, weights_hidden, weight_output

    def fit(self, x_train, y_train):
        weight_input, weights_hidden, weight_output, bias_hidden = self.initialize_weights()
        l_out_perfor = []
        for exa, y in zip(x_train, y_train):
            y_preds, hidden_layer_outputs = self.feed_forward_propagation(exa, weight_input, weights_hidden,
                                                                          weight_output, bias_hidden,
                                                                          self.activation_function)
            l_out_perfor.append(y_preds)
        for iteration in range(self.num_of_iterations):
            for exa, y in zip(x_train, y_train):
                # print("exa,y",exa,y)
                # Feed forward propagation
                y_pred, hidden_layer_output = self.feed_forward_propagation(exa, weight_input, weights_hidden,
                                                                            weight_output, bias_hidden,
                                                                            self.activation_function)
                # Back propagation
                weight_input, weights_hidden, weight_output = self.back_propagation(hidden_layer_output, y_pred, exa, y,
                                                                                    weight_input, weights_hidden,
                                                                                    weight_output, bias_hidden,
                                                                                    self.activation_function)
        l_out_perfor = []
        for exa, y in zip(x_train, y_train):
            y_preds, hidden_layer_outputs = self.feed_forward_propagation(exa, weight_input, weights_hidden,
                                                                          weight_output, bias_hidden,
                                                                          self.activation_function)
            l_out_perfor.append(y_preds)
        y_pred_arr = np.array(l_out_perfor)
        y_pred_arr = y_pred_arr.argmax(axis=1)
        y_train = np.argmax(y_train, axis=1)
        return weight_input, weights_hidden, weight_output, y_pred_arr

    def predict(self, x_test, y_test, weight_input, weights_hidden, weight_output):
        y_pred_list = []
        y_test_list = []
        for exa, y in zip(x_test, y_test):
            y_pred, hidden_layer_output = self.feed_forward_propagation(exa, weight_input,
                                                                        weights_hidden, weight_output,
                                                                        self.bias_hidden, self.activation_function)
            y_test = np.argmax(y_test)
            y_test_list.append(y_test)
            y_pred = np.argmax(y_pred)
            y_pred_list.append(y_pred)

        self.confusion_matrix(y_test_list, y_pred_list)
        return y_pred_list
