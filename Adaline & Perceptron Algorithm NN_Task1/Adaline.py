import numpy as np
class Adaline():
    def __init__(self, learning_rate=0.001, n_iters=1000,mse=0.01,bais = 0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.mse = mse
        self.weights = None
        self.bais = bais
        
    def fit(self,data, labels):
        X = np.array(data)
        y = np.array(labels)
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.costs = []
        for _ in range(self.n_iters):
            output = np.dot(X, self.weights[1:]) + (self.weights[0] if self.bais else 0)
            errors = (y - output)
            self.weights[1:] += self.lr *np.dot(X.T, errors)
            self.weights[0] += self.lr * errors.sum() if self.bais else 0
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)
            
            summation = 0
            for i ,val in enumerate(self.costs):
                summation+=val
            if summation <=self.mse:
                break
        return self

    def predict(self, X):
        y_predicted = []
        linear_output = np.dot(X, self.weights[1:]) + (self.weights[0] if self.bais else 0)
        for i in linear_output:
            y_predicted.append(self.activation_fun(i))
        return y_predicted
    
    def activation_fun(self, value):
        if value > 0:
            return 1
        else:
            return -1
        
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
        confusion_matrix = self. test_array_confusion_matrix(x,y)
        correct_predictions = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
        total_predictions = sum(sum(row) for row in confusion_matrix)
        return correct_predictions / total_predictions