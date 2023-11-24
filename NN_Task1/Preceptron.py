import numpy as np
class Perceptron():
    def __init__(self, learning_rate=0.0001, n_iters=1000,bias=0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bi = np.random.random(1)
        self.bias = bias

        
    def fit(self, data, labels):
        X = np.array(data)
        y = np.array(labels)
        self.weights = np.random.random(X.shape[1])
        for i in range(self.n_iters):
            for idx ,x_i in enumerate(X):
                output = self.activation_fun(np.dot(x_i,self.weights) + (self.bi if self.bias else 0))
                if output != y[idx]:
                    update = self.lr * (y[idx] - output)
                    self.weights += update * x_i
                    self.bi += (update if self.bias else 0)
            
                
    def activation_fun(self, value):
      if value >= 0:
        return 1
      else:
        return -1
        
    def predict(self, X):
        y_predicted = []
        linear_output = np.dot(X, self.weights) + (self.bi if self.bias else 0)
        for i in linear_output:
            y_predicted.append(self.activation_fun(i))
        return y_predicted
    
    
    def test_sample(self,x):
        return self.predict(x)

    def test_array_confusion_matrix(self,x,actual):
        predicted = self.predict(x)
        matrix = [[0 for _ in range(2)] for _ in range(2)]
        tn =0 
        fp = 0 
        fn = 0
        tp = 0
        for i,val in enumerate(actual):
          if val == -1 and predicted[i] == -1:
              tn += 1
          elif val  == -1 and predicted[i] == 1:
              fp += 1
          elif val  == 1 and predicted[i] == -1:
              fn += 1
          elif val  == 1 and predicted[i] == 1:
              tp += 1
              
          matrix[0][0] = tn
          matrix[0][1] = fp
          matrix[1][0] = fn
          matrix[1][1] = tp
        return matrix
        
    def accuracy(self,x,y):
        confusion_matrix = self.test_array_confusion_matrix(x,y)
        correct_predictions = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
        total_predictions = sum(sum(row) for row in confusion_matrix)
        return correct_predictions / total_predictions
    