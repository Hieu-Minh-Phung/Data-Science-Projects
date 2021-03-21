# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 23:56:18 2020

@author: Reagan Phung
"""
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def keras_model(file1, file2):
          
  data = pd.read_csv(file1, header = None)
  y_train = data[data.columns[-1]]
  x_train = data.drop(data.columns[-1], axis = 1)
  
  model = Sequential()
  cols = x_train.shape[1]
  model.add(Dense(80, activation = "relu", input_shape = (cols,)))
  model.add(Dense(150, activation = "relu"))
  model.add(Dense(50, activation = "relu"))
  model.add(Dense(30, activation = "relu"))
  model.add(Dense(1, activation = "sigmoid"))
  
  opt = Adam()
  model.compile(optimizer = opt, loss = "binary_crossentropy", metrics = ["accuracy"])
  history = model.fit(x_train, y_train, epochs = 100, batch_size = 50)
  
  data = pd.read_csv(file2, header = None)
  y_test = data[data.columns[-1]]
  x_test = data.drop(data.columns[-1], axis = 1)
  _, accuracy = model.evaluate(x_test, y_test, verbose=0)
  print('Accuracy: %.2f' % (accuracy*100))
  acc =  history.history['accuracy']
  loss =  history.history['loss']
  epochs = range(1, len(acc) + 1)
  plt.plot(epochs, acc, 'b', label=' accuracy')
  plt.plot(epochs, loss, 'r', label=' loss')
  plt.ylabel('accuracy/loss')
  plt.xlabel('epoch')
  plt.title('Training accuracy and loss')
  plt.show()
  
file1 = "p2_train.csv"
file2 = "p2_test.csv"

def main():
    keras_model(file1, file2)
main()