#! /usr/env python
# -*- coding: utf8 -*-


import math
import sys
import os
import numpy as np


class SVMClassifier:

    def __init__(self):
        self.source_path = "/home/marshao/SVMClassifier/"
        self.alphas = []    #list of langarange multipliers of Samples
        self.b = 0         #b for determine function
        self.TrainingSamples = []   #Matrix of Training Sample
        self.TrainingLabels = []  # Matrix of Training Labels
        self.CVSamples = []       #Matrix of CV samples
        self.CVLabels = []
        self.TestingSampels = []     #Matrix of test samples
        self.TestingLabels = []       #Marix of testing labels
        self.predicts = []  #Matrix of predicts
        self.T = 0.001               #Tolerance(Accuracy) of System
        self.C = 10                   #Penotal Coefficients
        self.step = 0.001             #Min step length of alpha1
        self.iter_times = 50           #Max iterration times
        self.Models_Dict = []           #Dictonary to Store Models

    def LoadData(self, model):
        '''Load Samples Data into Numpy Matrix
           Initial Parameters
        '''

        source = 'TrainingSamples.csv'
        fn = open(source, "r")
        for line in fn:
            line = line[:-2]  # Remove the /r/n
            vlist = line.split(",")
            self.TrainingSamples.append([vlist[0], vlist[1]])
            self.TrainingLabels.append(vlist[2])
        print "Loaded %s Training Samples" %len(self.TrainingSamples)
        fn.close()

        i = 0
        while i < len(self.TrainingSamples):
            self.alphas.append([0])
            i+=1
        print "Length of alpha is %s"%len(self.alphas)

        if model == 'Testing':
            fn = open('TestingSamples.csv', 'r')
            for line in fn:
                line = line[:-2]  # Remove the /r/n
                vlist = line.split(",")
                self.TestingSampels.append([vlist[0], vlist[1]])
                self.TestingLabels.append(vlist[2])
            print "Loaded %s Testing Samples" %len(self.TestingSampels)
            fn.close()
        elif model == 'CV':
            fn = open('CVSamples.csv', 'r')
            for line in fn:
                line = line[:-2]  # Remove the /r/n
                vlist = line.split(",")
                self.CVSamples.append([vlist[0], vlist[1]])
                self.CVLabels.append(vlist[2])
            print "Loaded %s CV Samples" %len(self.TestingSampels)
            fn.close()
        else:
            pass

        return

    def _Find_Alpha1(self):
        '''Find Alpha1 by checking the violation of KKT conditions'''
        pass

    def _Find_Alpha1_Non_boundary(self):
        '''Find Alpha1 from Non boundary samples'''
        pass

    def _Find_Alpha1_Whole(self):
        '''Find Alpha1 from Entire sample set'''
        pass

    def _Find_Alpha2(self):
        '''Find Alpha2 when Alpha1 is decided
        Step1: Find Alpha2 by Max|E1-E2|
        Step2: If (Alpha2New-Alpha2old)<Step, then find Alpha2 from boundary samples
        Step3: If (Alpha2New-Alpha2old)<Step, then find Alpha2 from all samples
        '''
        pass

    def _Cal_F(self, x2):
        '''The determine function F(x) to calculate the prediciton values
            x2 is a sample (vector of atributes Ex:[0.34, 0.21, 0.57])
        '''
        # Vectorize Smaples
        x1 = np.array(self.TrainingSamples)
        y = np.array(self.TrainingLabels)
        alpha = np.array(self.alphas)
        Kernal_Val = self._Cal_Kernal('l', x1, x2)
        print Kernal_Val
        print "Prediction Lable of data sample %s is %s"%(x2, val)

    def _Cal_Kernal(self, mode, x1, x2, delta=8, R=1, d=2):
        '''Kernal function to calculate Kernals
            Linear Kernal
            Gaussin Kernal RBF
            Polinominal Kernal
            Kernals are based on the Dot.Product of X-Predicting and X-Samples
        '''
        if mode == 'l':
            self._Cal_Linear_Kernal(x1,x2)
        elif mode == 'g':
            self._Cal_Gaussin_Kernal(x1,x2,delta)
        elif mode == 'p':
            self._Cal_Polinomial_Kernal(x1,x2,R,d)
        else:
            print "No such kernal type, kernal must be in l/g/p"

    def _Cal_Linear_Kernal(self, x1, x2):
        '''Calculation of linear Kernal
            Vector Solution: x1 and x2 are the sample vectors
        '''
        val = np.dot(x1, x2)
        print "Linear Kernal Value is %s"%val
        return val

    def _Cal_Gaussin_Kernal(self,x1,x2,delta):
        '''Calculation of Gaussion Kernal
            Vector Solution: x1 and x2 are the sample vectors
            delta is squared diviation with default value 8
        '''
        #Convert x1 and x2 into Numpy ndarray
        x1 = np.array(x1)
        x2 = np.array(x2)
        val = math.exp((-1)*(np.sum(np.square(x1-x2))/(2*delta*delta)))
        #val = math.exp((-1)*(math.pow(np.linalg.norm((x1-x2)),2)/(2*delta*delta)))
        print val
        return val

    def _Cal_Polinomial_Kernal(self,x1,x2,R,d):
        '''Calculation of Polinomial Kernal
        Vector Solution: x1 and x2 are the sample vectors
        R is the constant with default value 1
        d is the power degree with default value 2
        '''
        val = math.pow((self._Cal_Linear_Kernal(x1,x2)+R),d)
        print val
        return val


    def _Cal_Ei(self):
        '''Function to calculate difference between label and prediciton
        Yi - f(x)
        '''
        pass

    def _Cal_Alpha1(self):
        '''Function to renew Alpha1'''
        pass

    def _Cal_Alpha2(self):
        '''Function to renew Alpha2'''
        pass

    def _Cal_eta(self):
        '''Function to calcate eta'''
        pass

    def _Cal_b(self):
        '''Function to calculate b value when Alpha1 and Alpha2 are renewed'''
        pass

    def Train_Model(self):
        '''Function to train SVM models with training sample set'''
        pass

    def Cross_Validate_Model(self):
        '''Function to cross validate model with cv sample set'''
        pass

    def Test_Model(self):
        '''Function to Test models with test sample set'''
        pass

    def Predict(self):
        '''Function to use model to predict values'''
        pass

    def Performance_Diag(self):
        '''Function to evaluate performance of models'''
        pass

    def pause(self):
        programPause = raw_input("Press any key to continue")

def main():

    model = SVMClassifier()
    model.LoadData('Testing')

    x1 = [[1.,2.,3.,4.],[4.,5.,6.,7.]]
    x2 = [5.,6.,7.,8.]
    x1 = [1,1]
    x2 = [3,4]
    model._Cal_Kernal("l", x1,x2)

if __name__ == '__main__':
    main()