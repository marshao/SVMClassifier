#! /usr/env python
# -*- coding: utf8 -*-


import math
import sys
import time
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
        self.C = 10.0                   #Penotal Coefficients
        self.Step = 0.001             #Min step length of alpha1
        self.Loop = 20           #Max iterration times
        self.Models_Dict = []           #Dictonary to Store Models
        self.KernalType = 'l'          # predefine the Kernal Type is linear Kernal
        self.GaussinSigma = 8.0           # Define a default value of Delta for Gaussin Kernal
        self.PolinomailR = 1.0            # Default value of Polinomail Kernal R
        self.Polinomaild = 2.0            # Default value of Polinomail Kernal d
        self.alpha1Idx = 0                 #Index of Alpha 1
        self.alpha2Idx = 0                 #Index of Alpha 2

    def LoadData(self, model, source=None):
        '''Load Samples Data into Numpy Matrix
           Initial Parameters
        '''
        if source is None:
            source = 'TrainingSamples2.csv'
        fn = open(source, "r")
        for line in fn:
            line = line[:-1]  # Remove the /r/n
            vlist = line.split(",")
            self.TrainingSamples.append([float(vlist[0]), float(vlist[1])])
            self.TrainingLabels.append(float(vlist[2]))
        print "Loaded %s Training Samples" %len(self.TrainingSamples)
        fn.close()

        # Initialize Alpha list
        alpha = float(0.0)
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

    def _Update_Variables(self, idx1=None, alpha1=None, idx2=None, alpha2=None, b=None, T=None, C=None, Step=None, Loop=None, Sigma=None):
        '''Update Variables
            idx1: Alpha1
            idx2: Alpha2
            b
            T
            C
            Step
            Loop
        '''
        if idx1 != None and alpha1 != None:
            self.alphas[idx1] = alpha1
        if idx2 != None and alpha2 != None:
            self.alphas[idx2] = alpha2
        if b != None:
            self.b = b
        if T != None:
            self.T = T
        if C != None:
            self.C = C
        if Step != None:
            self.Step = Step
        if Loop != None:
            self.Loop = Loop
        if Sigma != None:
            self.GaussinSigma = Sigma

    def _Find_Alpha2(self, idx1, E1, KernalType = None):
        '''Find Alpha2 when Alpha1 is decided
        Step1: Find Alpha2 by Max|E1-E2|
        Step2: If (Alpha2New-Alpha2old)<Step, then find Alpha2 from boundary samples
        Step3: If (Alpha2New-Alpha2old)<Step, then find Alpha2 from all samples
        '''
        if KernalType is None:
            KernalType = self.KernalType
        found, alpha2, eta = self._Find_Alpha2_Nonbound(idx1, E1, KernalType)
        if found == False:
            found,alpha2,eta = self._Find_Alpha2_MaxE(idx1, E1, KernalType)
        return found, alpha2, eta

    def _Find_Alpha2_MaxE(self, idx1, E1, KernalType = None):
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
        eta = 0
        found = False
        for idx2 in range(len(self.alphas)):
            if alpha1 == idx2: continue
            # eta is the distance of two Alphas, it must be > 0. If eta < 0, then the eta must be wrong
            eta_tmp = self._Cal_eta(idx1, idx2, KernalType)
            if eta_tmp < 0:
                print "   eta %s should not be less than 0." % eta_tmp
                continue
            E2 = self._Cal_Ei(idx2, KernalType)
            tmp = math.fabs(E1-E2)
            if tmp > gap:
                gap = tmp
                alpha2 = idx2
                eta = eta_tmp
                found = True
        if found:print "For Alpha1 at %s the best Alpha2 is %s"%(idx1, alpha2)
        return found, alpha2, eta

    def _Find_Alpha2_Nonbound(self, idx1, E1, KernalType=None):
        '''Find the alpha2 with Max |E1-E2| and which value is not the boudary value, boundary value is [0-C]'''
        if KernalType is None:
            KernalType = self.KernalType

        alpha1 = idx1
        alpha2 = 0
        # gap is |E1 - E2|
        gap = 0
        eta = 0
        found = False
        for idx2 in range(len(self.alphas)):
            # Two alpha should not be same
            if alpha1 == idx2: continue
            # Alpha2 should not have the boundary value
            if self.alphas[alpha2] == 0 or self.alphas[alpha2] == self.C:continue
            # eta is the distance of two Alphas, it must be > 0. If eta < 0, then the eta must be wrong
            eta_tmp = self._Cal_eta(idx1, idx2, KernalType)
            if eta_tmp < 0:
                print "   eta %s should not be less than 0." % eta_tmp
                continue
            E2 = self._Cal_Ei(idx2, KernalType)
            tmp = math.fabs(E1-E2)
            if tmp > gap:
                gap = tmp
                alpha2 = idx2
                eta = eta_tmp
                found = True
        if found:print "For Alpha1 at %s the best Alpha2 is %s"%(idx1, alpha2)
        return found, alpha2, eta

    def _Cal_F(self, x2, KernalType=None):
        '''The determine function F(x) to calculate the prediciton values
            x2 is the sample (vector of atributes Ex:[0.34, 0.21, 0.57]) waiting to be predicted
        '''
        # Setting the default Kernal type
        if KernalType is None:
            KernalType = self.KernalType

        # Calculate Kernal, return value is a ndarray
        x1 = self.TrainingSamples
        print "   x2 is :%s"%x2
        Kernal_Val = self._Cal_Kernal(x1, x2, KernalType)
        print "   Kernal_Val is %s"%Kernal_Val
        #print type(val)
        #Kernal_Val = np.ndarray(val)

        #Vectorize label y and alpha into ndarry for the next step of calculation
        y = np.array(self.TrainingLabels)
        alpha = np.array(self.alphas)
        prediction_value = sum(alpha*y*Kernal_Val) + self.b
        #print "Prediction Lable of data sample %s is %s"%(x2, prediction_value)
        return prediction_value

    def _Cal_Kernal(self,  x1, x2, mode=None,sigma=None, R=None, d=None):
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
        if sigma is None:
            sigma = self.GaussinSigma
        if R is None:
            R = self.PolinomailR
        if d is None:
            d = self.Polinomaild

        if mode == 'l':
            return self._Cal_Linear_Kernal(x1,x2)
        elif mode == 'g':
            return self._Cal_Gaussin_Kernal(x1,x2,sigma)
        elif mode == 'p':
            return self._Cal_Polinomial_Kernal(x1,x2,R,d)
        else:
            print "No such kernal type, kernal must be in l/g/p"

    def _Cal_Linear_Kernal(self, x1, x2):
        '''Calculation of linear Kernal
            Vector Solution: x1 and x2 are the sample vectors
        '''
        val = np.dot(x1, x2)
        #print "Linear Kernal Value is %s"%val
        return val

    def _Cal_Gaussin_Kernal(self,x1,x2,sigma):
        '''Calculation of Gaussion Kernal
            Vector Solution: x1 and x2 are the sample vectors
            Sigma is squared diviation with default value 8
        '''
        #Convert x1 and x2 into Numpy ndarray
        x1 = np.array(x1)
        x2 = np.array(x2)
        #print "   x1-x2 is:%s"%(x1-x2)
        #print "   np.square(x1-x2) is:%s"%np.square(x1-x2)
        #print "   np.sum(np.square(x1-x2)) is :%s"%np.sum(np.square(x1-x2))
        #val = (np.sum(np.square(x1-x2))/(2*sigma*sigma))
        #print "   np.sum(np.square(x1-x2))/(2*delta*delta)) is:%s"%val
        val = math.exp((-1)*(np.sum(np.square(x1-x2))/(2*sigma*sigma)))
        #print "   math.exp((-1)*(np.sum(np.square(x1-x2))/(2*delta*delta))) is :%s"%val
        #val = math.exp((-1)*(math.pow(np.linalg.norm((x1-x2)),2)/(2*delta*delta)))
        #print val
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
        print "   idx is %s, and x2 is: %s, predict is :%s, Eidx is : %s" % (idx, x2, predict, Eidx)
        #print "Eidx for training sample %s is %s"%(idx, Eidx)
        return Eidx

    def _Cal_Alpha1(self, idx1, idx2, new_alpha2):
        '''Function to renew Alpha1
            idx1 and idx 2 are the old alpha1 and alpha2 index
            new_alpha2 is the new alpha2 value
        '''
        y1 = self.TrainingLabels[idx1]
        y2 = self.TrainingLabels[idx2]
        old_alpha1 = self.alphas[idx1]
        old_alpha2 = self.alphas[idx2]

        new_alpha1 = old_alpha1 + y1*y2*(old_alpha2-new_alpha2)
        #print "New Alpha1 %s value is %s"%(idx1, new_alpha1)
        return new_alpha1

    def _Cal_Alpha2(self, idx1, idx2, E1, eta, C=None, Step=None, KernalType=None):
        '''Function to renew Alpha2
            idx1 is the index of old Alpha1
            idx2 is the index of old Alpha2
            E1 is the Ei value for alpha with index idx1
            eta is the eta value for Alpha1 and Alpha2
            L and H are the low and high border of new Alpha2 value
            C is the penalty coefficient
            Step is min change of Alpha2
        '''
        if KernalType is None:
            KernalType = self.KernalType
        if C is None:
            C = self.C
        if Step is None:
            Step = self.Step

        y1 = self.TrainingLabels[idx1]
        y2 = self.TrainingLabels[idx2]
        # Get the value of Alpha
        old_alpha1 = self.alphas[idx1]
        old_alpha2 = self.alphas[idx2]
        L = 0.0
        H = 0.0
        valid = True

        #Calculate new alpha2
        E2 = self._Cal_Ei(idx2,KernalType)
        print "   E1-E2=%s,  eta = %s, y2*(E1-E2)=%s, y2*(E1-E2)/eta=%s"%((E1-E2), eta, (y2*(E1-E2)),(y2*(E1-E2)/eta))
        new_alpha2 = old_alpha2 + y2 * (E1 - E2) / eta

        # Set boundary for new_alpha2_value value
        if y1 == y2:
            L = max(0.0, (old_alpha2+old_alpha1-C))
            H = min(C, (old_alpha2+old_alpha1))
        else:
            L = max(0.0, (old_alpha2-old_alpha1))
            H = min(C, (C+old_alpha1+old_alpha2))

        # Clip new_alpha2_value value
        if new_alpha2 > H: clipped_new_alpha2 = H
        elif new_alpha2 < L: clipped_new_alpha2 = L
        else: clipped_new_alpha2 = new_alpha2

        print "   For Alpha2 at index %s, old value is %s, new value is %s, clipped value is %s, L is %s, H is %s"%(idx2, old_alpha2, new_alpha2, clipped_new_alpha2, L, H)
        # Vrifile whether the clipped_new_alpha2 moved big enough > step
        if abs(clipped_new_alpha2 - old_alpha2) < Step:
            print "   Old_Alpha2:Value %s:%s - clipped_New_Alpha2:%s = Diff %s cannot provide enough change to old Alpha2"%(idx2, old_alpha2, clipped_new_alpha2, abs(clipped_new_alpha2 - old_alpha2))
            valid = False

        return valid, clipped_new_alpha2

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
        #print "eta value for x1 %s and x2 %s is %s"%(idx1, idx2, eta)
        return eta

    def _Cal_b(self, idx1, idx2, new_alpha1, new_alpha2, KernalType = None, C=None):
        '''Function to calculate b value when Alpha1 and Alpha2 are renewed'''
        if KernalType is None:
            KernalType = self.KernalType
        if C is None:
            C = self.C

        old_b = self.b
        old_alpha1 = self.alphas[idx1]
        old_alpha2 = self.alphas[idx2]
        y1 = self.TrainingLabels[idx1]
        y2 = self.TrainingLabels[idx2]
        K11 = self._Cal_Kernal(old_alpha1,old_alpha1,KernalType)
        K12 = self._Cal_Kernal(old_alpha1,old_alpha2,KernalType)
        K22 = self._Cal_Kernal(old_alpha2,old_alpha2,KernalType)
        K21 = self._Cal_Kernal(old_alpha2,old_alpha1,KernalType)
        E1 = self._Cal_Ei(idx1, KernalType)
        E2 = self._Cal_Ei(idx2, KernalType)

        # Calculate b
        new_b1 = (old_alpha1 - new_alpha1) * y1 * K11 + (old_alpha2 - new_alpha2) * y2 * K21 - E1 + old_b
        new_b2 = (old_alpha1 - new_alpha1) * y1 * K12 + (old_alpha2 - new_alpha2) * y2 * K22 - E2 + old_b
        if new_alpha1 > 0 and new_alpha1 < C: new_b = new_alpha1
        elif new_alpha2 > 0 and new_alpha2 < C: new_b = new_alpha2
        else:   new_b = (new_b1 + new_b2)/2
        print "   new b is %s"%new_b
        return new_b



    def Train_Model(self, C = None, T=None, Loop = None, KernalType = None, Step=None, Sigma=None):
        '''Function to train SVM models with training sample set
            C is penalty coefficient with Default value self.C = 10
            T is Tolerance coefficient with Default value self.T=0.001
            Loop is max times of iteration with Default value self.Loop = 20
            Step is the minimal change of alpha with default value self.Step = 0.001
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
        if Step is None:
            Step = self.Step
        if Sigma is None:
            sigma = self.GaussinSigma
        # Initiate the index of alpha1_idx and alpha2_idx as 0
        alpha1_idx = 0
        alpha2_idx = 0
        old_alpha1 = self.alphas[alpha1_idx]
        old_alpha2 = self.alphas[alpha2_idx]

        #Loop the list of Alpha until the reach the max loop number or all alphas have no furthur change
        iter = 0
        while iter < Loop:
            iter += 1
            done = False # Parameter to determine whether the training is finsihed or not.
            print "Iteration %s is start"%iter
            print"--------------------------------------------------------------------"
            time.sleep(4)
            done = True
            # find alpha1_idx's index by checking the violation of KKT condition
            for alpha1_idx in range(len(self.alphas)):
                y1 = self.TrainingLabels[alpha1_idx]
                E1 = self._Cal_Ei(alpha1_idx, KernalType)
                print "For Alpha1 at %s with value %s, label y1 is %s, E1 is %s, C is %s, T is %s, alpha1 value is %s, y1*E1 = %s"%(alpha1_idx, old_alpha1, y1, E1, C, T, old_alpha1, y1*E1)
                # KKT condition vilation checking
                if (old_alpha1 < C and (y1*E1)< -T) or (old_alpha1 > 0 and (y1*E1) > T):
                    done = False
                    #Find the most proper alpha2_idx's index
                    found, alpha2_idx, eta = self._Find_Alpha2(alpha1_idx, E1, KernalType)
                    if found == False:
                        # Cannot find proper Alpha2 for Alpha1, change to an new alpha1_idx
                        print "   Cannot find proper Alpha2 for Alpha1 %s, change to an new Alpha1" % alpha1_idx
                        continue

                    # Calculate new alpha2_idx
                    valid, clipped_new_alpha2 = self._Cal_Alpha2(alpha1_idx, alpha2_idx, E1, eta, C, Step, KernalType)
                    if valid == False:
                        #print "There is no valid value for alpha2 at index %s"%alpha2_idx
                        continue

                    # Calculate new alpha1_idx
                    new_alpha1 = self._Cal_Alpha1(alpha1_idx, alpha2_idx, clipped_new_alpha2)

                    # Calculate b when new alpha1 and alpha2 be calculated out
                    new_b= self._Cal_b(alpha1_idx,alpha2_idx, new_alpha1, clipped_new_alpha2, KernalType, C)

                    # update variables
                    self._Update_Variables(idx1=alpha1_idx, alpha1=new_alpha1,
                                           idx2=alpha2_idx, alpha2= clipped_new_alpha2,
                                           b = new_b)
                    print "   Loop:%s, Alpha1:idx1 = %s:%s; Alpha2:idx2 = %s:%s; Updated to:  Alpha1_New:%s and Alpha2_New:%s and b_new:%s"%(iter,alpha1_idx, old_alpha1, alpha2_idx, old_alpha2, new_alpha1, clipped_new_alpha2,new_b)

            # Count the undated Alpha for each iteration
            j = 0
            for alpha in range(len(self.alphas)):
                if self.alphas[alpha] != 0: j+=1
            print "In Iteraton %s, %s alphas has been updated" %(iter, j)
            print self.alphas
            print "____________________________________________________"
            print ""
            if done:
                break

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
    model.LoadData('Train', 'TrainingSamples3.csv')

    x1 = [[1.,2.],[4.,5.]]
    #print x1
    #print np.sum(x1)
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
    #model._Cal_eta(12,78,'l')
    model._Update_Variables(C=2, Sigma=0.1)
    model.Train_Model(Loop=5, KernalType='g',)
    #print x1*x2


if __name__ == '__main__':
    main()