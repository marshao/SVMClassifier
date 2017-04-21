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
        self.Loop = 20           #Max iterration times
        self.Models_Dict = []           #Dictonary to Store Models
        self.KernalType = 'l'          # predefine the Kernal Type is linear Kernal
        self.GaussinDelta = 8           # Define a default value of Delta for Gaussin Kernal
        self.PolinomailR = 1            # Default value of Polinomail Kernal R
        self.Polinomaild = 2            # Default value of Polinomail Kernal d
        self.alpha1Idx = 0                 #Index of Alpha 1
        self.alpha2Idx = 0                 #Index of Alpha 2

    def LoadData(self, model):
        '''Load Samples Data into Numpy Matrix
           Initial Parameters
        '''

        source = 'TrainingSamples.csv'
        fn = open(source, "r")
        for line in fn:
            line = line[:-1]  # Remove the /r/n
            vlist = line.split(",")
            self.TrainingSamples.append([float(vlist[0]), float(vlist[1])])
            self.TrainingLabels.append(float(vlist[2]))
        print "Loaded %s Training Samples" %len(self.TrainingSamples)
        fn.close()

        # Initialize Alpha list
        alpha = float(1.1)
        i = 0
        while i < len(self.TrainingSamples):
            self.alphas.append(alpha)
            i+=1
        print "Length of alpha is %s"%len(self.alphas)

        if model == 'Testing':
            fn = open('TestingSamples.csv', 'r')
            for line in fn:
                line = line[:-1]  # Remove the /r/n
                vlist = line.split(",")
                self.TestingSampels.append([float(vlist[0]), float(vlist[1])])
                self.TestingLabels.append(float(vlist[2]))
            print "Loaded %s Testing Samples" %len(self.TestingSampels)
            fn.close()
        elif model == 'CV':
            fn = open('CVSamples.csv', 'r')
            for line in fn:
                line = line[:-1]  # Remove the /r/n
                vlist = line.split(",")
                self.CVSamples.append([float(vlist[0]), float(vlist[1])])
                self.CVLabels.append(float(vlist[2]))
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

    def _Find_Alpha2(self, idx1, E1, KernalType = None):
        '''Find Alpha2 when Alpha1 is decided
        Step1: Find Alpha2 by Max|E1-E2|
        Step2: If (Alpha2New-Alpha2old)<Step, then find Alpha2 from boundary samples
        Step3: If (Alpha2New-Alpha2old)<Step, then find Alpha2 from all samples
        '''
        if KernalType is None:
            KernalType = self.KernalType

        alpha1 = idx1
        alpha2 = 0
        # gap is |E1 - E2|
        gap = 0
        for idx2 in range(len(self.alphas)):
            if alpha1 == idx2: continue
            # eta is the distance of two Alphas, it must be > 0. If eta < 0, then the eta must be wrong
            eta = self._Cal_eta(alpha1, idx2, KernalType)
            if eta < 0:
                print "eta %s should not be less than 0." % eta
                continue
            E2 = self._Cal_Ei(idx2, KernalType)
            tmp = math.fabs(E1-E2)
            if tmp > gap:
                gap = tmp
                alpha2 = idx2
        return alpha2


        pass

    def _Cal_F(self, x2, KernalType=None):
        '''The determine function F(x) to calculate the prediciton values
            x2 is the sample (vector of atributes Ex:[0.34, 0.21, 0.57]) waiting to be predicted
        '''
        # Setting the default Kernal type
        if KernalType is None:
            KernalType = self.KernalType

        # Calculate Kernal, return value is a ndarray
        x1 = self.TrainingSamples
        Kernal_Val = np.ndarray(self._Cal_Kernal(x1, x2, KernalType))

        #Vectorize label y and alpha into ndarry for the next step of calculation
        y = np.array(self.TrainingLabels)
        alpha = np.array(self.alphas)
        prediction_value = sum(alpha*y*Kernal_Val) + self.b
        print "Prediction Lable of data sample %s is %s"%(x2, prediction_value)
        return prediction_value

    def _Cal_Kernal(self,  x1, x2, mode=None,delta=None, R=None, d=None):
        '''Kernal function to calculate Kernals
            Linear Kernal
            Gaussin Kernal RBF
            Polinominal Kernal
            Kernals are based on the Dot.Product of X-Predicting and X-Samples
            x1 and x2 are the sample value
            x1 and x2 should be type of python list, not vector.
        '''
        if mode is None:
            mode=self.KernalType
        if delta is None:
            delta = self.GaussinDelta
        if R is None:
            R = self.PolinomailR
        if d is None:
            d = self.Polinomaild

        if mode == 'l':
            return self._Cal_Linear_Kernal(x1,x2)
        elif mode == 'g':
            return self._Cal_Gaussin_Kernal(x1,x2,delta)
        elif mode == 'p':
            return self._Cal_Polinomial_Kernal(x1,x2,R,d)
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


    def _Cal_Ei(self,idx, KernalType=None):
        '''Function to calculate difference between label and prediction
        E_idx=f(x_idx)-y_idx
        idx is the index of sample vector waiting for be predicted
        Kernal_Type is the type of Kernal in Using
        Will this E1 function only usful when Training, otherwise the idx may be ambigous to pointing to sample in Training set or Test Set.Lets make is only for Tranining set first
        '''
        # Define the Kernal Type
        if KernalType is None:
            KernalType = self.KernalType

        # x2 is the sample vector waiting for be predicted.
        x2 = self.TrainingSamples[idx]
        predict = self._Cal_F(x2, KernalType)
        Eidx = predict - self.TrainingLabels[idx]
        print "Eidx for training sample %s is %s"%(idx, Eidx)
        return Eidx

    def _Cal_Alpha1(self):
        '''Function to renew Alpha1'''
        pass

    def _Cal_Alpha2(self, idx2):
        '''Function to renew Alpha2
            idx2 is the index of old Alpha2
        '''
        pass

    def _Cal_eta(self, idx1, idx2, KernalType):
        '''Function to calcate eta
        eta = K11+K22-2K12
        K is Knernal,
        idx1 and idx2 is the index of Training Samples, also correspond to the index of Alpha
        Kernal Type is the Kernal Type
        Presume this eta Calculaton will only be using in Training Set
        '''
        # Define the default Kernal Type
        if KernalType is None:
            KernalType = self.KernalType

        x1 = self.TrainingSamples[idx1]
        x2 = self.TrainingSamples[idx2]
        K11 = self._Cal_Kernal(x1,x1,KernalType)
        K22 = self._Cal_Kernal(x2,x2,KernalType)
        K12 = self._Cal_Kernal(x1,x2,KernalType)
        eta = K11+K22-2*K12
        print "eta value is %s"%eta
        return eta

    def _Cal_b(self):
        '''Function to calculate b value when Alpha1 and Alpha2 are renewed'''
        pass

    def Train_Model(self, C = None, T=None, Loop = None, KernalType = None):
        '''Function to train SVM models with training sample set
            C is penalty coefficient with Default value self.C = 10
            T is Tolerance coefficient with Default value self.T=0.001
            Loop is max times of iteration with Default value self.Loop = 20
        '''
        # Setting Default Parameters
        if C is None:
            C = self.C
        if T is None:
            T = self.T
        if Loop is None:
            Loop = self.Loop
        if KernalType is None:
            KernalType = self.KernalType
        # Initiate the index of alpha1 and alpha2 as 0
        alpha1 = 0
        alpha2 = 0

        #Loop the list of Alpha until the reach the max loop number or all alphas have no furthur change
        iter = 0
        while iter < Loop:
            iter += 1
            # find alpha1's index by checking the violation of KKT condition
            for alpha1 in range(len(self.alphas)):
                y1 = self.TrainingLabels[alpha1]
                E1 = self._Cal_Ei(alpha1, KernalType)
                # KKT condition vilation checking
                if (self.alphas[alpha1] < C and (y1*E1)< -T) or (self.alphas[alpha1] > 0 and (y1*E1) > T):

                    #Find the most proper alpha2's index
                    alpha2 = self._Find_Alpha2(alpha1, E1, KernalType)

                    # Calculate new alpha2
                    new_alpha2 = self._Cal_Alpha2(alpha2)



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
    model.LoadData('Train')

    x1 = [[1.,2.,3.,4.],[4.,5.,6.,7.]]
    x2 = [5.,6.,7.,8.]
    #x1 = np.array([2,2,6,3])
    #x2 = np.array([3,4,2,5])
    #x1 = [2.,2.,6.,3.]
    #x2 = [0.7,4.,2.,5.]
    x1 = model.TrainingSamples
    x2 =[0.77, 0.42]
    #model._Cal_Kernal("l", x1,x2)
    #model._Cal_F(x2,'l')
    #model._Cal_Ei(25)
    model._Cal_eta(12,78,'l')
    #print x1*x2


if __name__ == '__main__':
    main()