import tkinter as tk
from Preprocess import * 
from Preceptron import *
from Adaline import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



root = tk.Tk()
fig , ax =plt.subplots()
canvas = FigureCanvasTkAgg(fig , master = root)
canvas.get_tk_widget().place(x = 0.1,y=310)
root.title("Task 1")
root.geometry("1000x800")

##Feature 1 
def Feature1(event):
  return selected_feature1.get()
label = tk.Label(root,text='Feature 1',font=('Arial',15))
label.place(relx=0.01,rely=0.01)
selected_feature1 = tk.StringVar()
selected_feature1.set('Area')
dropdown = tk.OptionMenu(root, selected_feature1,'Area', "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes",command=Feature1)
dropdown.place(relx=0.1,rely=0.01)
dropdown.configure(font=('Arial',12))


##Feature 2
def Feature2(event):
  return selected_feature2.get()
label = tk.Label(root,text='Feature 2',font=('Arial',15))
label.place(relx=0.3,rely=0.01)
selected_feature2 = tk.StringVar()
selected_feature2.set('Perimeter')
dropdown = tk.OptionMenu(root, selected_feature2,'Area', "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes",command=Feature2)
dropdown.place(relx=0.4,rely=0.01)
dropdown.configure(font=('Arial',12))


##Class 1 
def Class1(event):
  return selected_Class1.get()
label = tk.Label(root,text='Class 2',font=('Arial',15))
label.place(relx=0.01,rely=0.1)
selected_Class1 = tk.StringVar()
selected_Class1.set('BOMBAY')
dropdown = tk.OptionMenu(root, selected_Class1,'BOMBAY', "CALI", "SIRA",command=Class1)
dropdown.place(relx=0.1,rely=0.1)
dropdown.configure(font=('Arial',12))


##Class 2
def Class2(event):
  return selected_Class2.get()
label = tk.Label(root,text='Class 2',font=('Arial',15))
label.place(relx=0.3,rely=0.1)
selected_Class2 = tk.StringVar()
selected_Class2.set('CALI')
dropdown = tk.OptionMenu(root, selected_Class2,'BOMBAY', "CALI", "SIRA",command=Class2)
dropdown.place(relx=0.4,rely=0.1)
dropdown.configure(font=('Arial',12))



##Bias
def Bias():
  return var.get()
  
var =tk.IntVar()
check = tk.Checkbutton(root ,text='Bias',variable=var, onvalue=1, offvalue=0 ,command=Bias)
check.configure(font=('Arial',15))
check.place(relx=0.6,rely=0.01)

##Choose Algorithm
radio = tk.IntVar()
def Algorithm():
  return radio.get()
  
R1 = tk.Radiobutton(root, text="Perceptron",variable=radio, value=1,command=Algorithm)
R1.configure(font=('Arial',15))
R1.place(relx=0.6,rely=0.1)
R2 = tk.Radiobutton(root, text="Adaline",variable=radio, value=2,command=Algorithm)
R2.configure(font=('Arial',15))
R2.place(relx=0.75,rely=0.1)

## Learning rate
def LearningRate(event):
  return float(entry1.get())
  
label = tk.Label(root,text='Learing rate',font=('Arial',13))
label.place(relx=0.01,rely=0.2)
entry1= tk.Entry(root, width= 20,font=('Arial',13))
entry1.focus_set()
entry1.place(relx=0.12,rely=0.2)


## NUMBER OF epochs
def Epochs(event):
  return int(entry2.get())

label = tk.Label(root,text='Number of Epochs',font=('Arial',13))
label.place(relx=0.3,rely=0.2)
entry2= tk.Entry(root, width= 20,font=('Arial',13))
entry2.place(relx=0.45,rely=0.2)


## MSE Threshold
def MSE(event):
  return float(entry3.get())
  
label = tk.Label(root,text='MSE Threshold',font=('Arial',13))
label.place(relx=0.65,rely=0.2)
entry3= tk.Entry(root, width= 20,font=('Arial',13))
entry3.place(relx=0.78,rely=0.2)

##Save data
def saveData(event):
  prep = preprocessing(Feature1(event), Feature2(event), Class1(event), Class2(event))
  X_train, X_test, y_train , y_test = prep.splitdata()
  if  int(Algorithm()) == 1:
    obj = Perceptron(LearningRate(event), Epochs(event),int(Bias()))
  elif int(Algorithm()) == 2:
    obj = Adaline(LearningRate(event), Epochs(event),MSE(event), int(Bias()))
    
  obj.fit(X_train, y_train)
  tk.Label(root,text=str(obj.accuracy(X_test,y_test)*100) +'%',font=('bold',17)).place(relx=0.79,rely=0.45)
  matrix = obj.test_array_confusion_matrix(X_test,y_test)
  printMatrix(matrix)
  
  newdata  = X_train.copy()
  newdata['Class'] = y_train.copy()
  data1= newdata[newdata['Class']==-1]
  data2= newdata[newdata['Class']==1]
  plot(data1,data2,X_train,obj,event)
  

def printMatrix(matrix):
  label = tk.Label(root,text=matrix[0][0],font=('bold',12))
  label.place(relx=0.8,rely=0.71)
  label = tk.Label(root,text=matrix[0][1],font=('bold',12))
  label.place(relx=0.88,rely=0.71)

  label = tk.Label(root,text=matrix[1][0],font=('bold',12))
  label.place(relx=0.8,rely=0.79)
  label = tk.Label(root,text=matrix[1][1],font=('bold',12))
  label.place(relx=0.88,rely=0.79)
  
def plot(ClassA, ClassB,X_train,obj,event):
  plt.scatter(ClassA[Feature1(event)],ClassA[Feature2(event)],c='red',s=60,label=Class1(event))
  plt.scatter(ClassB[Feature1(event)],ClassB[Feature2(event)],c='blue',s=60,label=Class2(event))  
  
  X = [np.min(X_train[Feature1(event)]), np.max(X_train[Feature1(event)])]
  X = np.array(X)
  if Algorithm() == 1:
    Y = -(obj.weights[0]* X + obj.bi) / obj.weights[1]
  else:
    Y = -(obj.weights[1]* X + obj.weights[0]) / obj.weights[2]
    
  plt.plot(X, Y, c='green', label=('Perceptron' if Algorithm() == 1 else 'Adaline'))

  plt.legend(loc = 'lower right', bbox_to_anchor=(1, 0))

  canvas.draw()
  

save = tk.Button(root, text= "Save data",font=('Arial',15))
save.place(relx= .45, rely= .3)
save.bind('<Button>',saveData)


##Accuracy && Confusion Matrix
label = tk.Label(root,text='Accuarcy',font=('bold',17))
label.place(relx=0.78,rely=0.4)
label = tk.Label(root,text='Confusion Matrix',font=('bold',17))
label.place(relx=0.74,rely=0.6)
label = tk.Label(root,text='Predicted Class',font=('bold',12))
label.place(relx=0.78,rely=0.65)
label = tk.Label(root,text='Actual Class',font=('bold',12))
label.place(relx=0.65,rely=0.75)

label = tk.Label(root,text='-1',font=('bold',12))
label.place(relx=0.8,rely=0.68)
label = tk.Label(root,text='1',font=('bold',12))
label.place(relx=0.88,rely=0.68)

label = tk.Label(root,text='-1',font=('bold',12))
label.place(relx=0.75,rely=0.71)
label = tk.Label(root,text='1',font=('bold',12))
label.place(relx=0.75,rely=0.79)


root.mainloop()