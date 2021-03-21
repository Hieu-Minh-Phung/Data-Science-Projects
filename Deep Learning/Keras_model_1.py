# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:21:29 2020

@author: Reagan Phung
"""

import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

def keras_model(file1, file2):
          
  data = pd.read_csv(file1, header = None)
  y_train = to_categorical(data[data.columns[-1]])
  x_train = data.drop(data.columns[-1], axis = 1)
  
  model = Sequential()
  cols = x_train.shape[1]
  model.add(Dense(50, activation = "relu", input_shape = (cols,)))
  model.add(Dense(30, activation = "softmax"))
  model.add(Dense(3, activation = "sigmoid"))
  
  opt = Adam()
  model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])
  history = model.fit(x_train, y_train, epochs = 200, batch_size = 10)
  
  data = pd.read_csv(file2, header = None)
  y_test = to_categorical(data[data.columns[-1]])
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

file1 = "p1_train.csv"
file2 = "p1_test.csv"

def main():
    keras_model(file1, file2)
main()