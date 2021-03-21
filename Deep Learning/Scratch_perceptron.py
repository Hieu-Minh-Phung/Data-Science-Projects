# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 06:07:52 2020

@author: ReaganPhung
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readfile(filename):
    file = pd.read_csv(filename, header = None)
    file = file.values
    return file

def filterfile(data):
    X = data[..., 0:2]
    Y = data[..., -1]
    return X,Y

class perceptron:
    def __init__(self, X, Y, lr):
         self.X = np.array(X, dtype = float)
         self.Y = np.array(Y)
         self.epoch = 1000
         self.yhat_arr = np.ones(len(Y))
         self.errors = np.ones(len(Y))
         self.learning_rate = float(lr)
         self.W = np.zeros(len(self.X[0]))
    
    def perceptron_train(self):
        
        n = 0                        
        loss = []                        
        while n < self.epoch: 
            for i in range(0,len(self.X)):                 
                activation = np.dot(self.X[i], self.W)   
                if activation >= 0:                               
                    yhat = 1                               
                else:                                   
                    yhat = -1
                self.yhat_arr[i] = yhat
            
                if yhat != self.Y[i]:
                    for j in range(0, len(self.W)):             
                        self.W[j] = self.W[j] + self.learning_rate*self.Y[i]*self.X[i][j]
            
            n += 1
            for i in range(0,len(self.Y)):     
                self.errors[i] = (self.Y[i] - self.yhat_arr[i])**2
            loss.append(0.5*np.sum(self.errors))
        return self.W, loss
        
    
    def perceptron_test(self):
        y_pred = []
        for i in range(0, len(self.X)):
            f = np.dot(self.X[i], self.Y)
            if f >= 0:                               
                yhat = 1                               
            else:                                   
                yhat = -1
            y_pred.append(yhat)
        
        return y_pred
        
filename1 = "train.csv"
filename2 = "test.csv"

def main():
    train = readfile(filename1)
    test = readfile(filename2)
    X_train, Y_train = filterfile(train)
    X_test, Y_test = filterfile(test)
    plt.title("The point in traning set: ")
    plt.scatter(train[...,0], train[...,1], c = train[...,2])
    A = perceptron(X_train, Y_train, 0.03)
    W = A.perceptron_train()[0]
    loss = A.perceptron_train()[1]
    B = perceptron(X_test, W, 0.03)
    y_pred = B.perceptron_test()
    
    # Different learning rate
    # C = perceptron(X_test, W, 10)
    # y_pred = C.perceptron_test()
    
    miss = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] != Y_test[i]: 
            miss += 1
            test[i][2] = 2
    
    print ("The accuracy is:", (len(Y_test) - miss)/len(Y_test), "with",miss, "missclassified points")

    plt.title("The point and separating line in the test set:")
    plt.scatter(test[...,0], test[...,1], c = test[...,2])
    X_1 = -10
    Y_1 = (-W[0] - W[0]*-10)/W[1]
    X_2 = 10
    Y_2 = (-W[0] - W[0]*10)/W[1]
    plt.plot([X_1, X_2], [Y_1, Y_2])
    plt.show()
    
main()