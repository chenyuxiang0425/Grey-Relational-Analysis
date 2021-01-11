# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

class GRA_Model():
    '''Grey relation analysis model'''
    def __init__(self,inputData,p=0.5):
        '''
        //Initialization parameters
        inputData: Input matrix, vertical axis is attribute name, first column is parent sequence
        p: Resolution coefficient, range 0~1，Generally take 0.5，The smaller the correlation coefficient is, the greater the difference is, and the stronger the discrimination ability is
        standard: Need standardization
        '''
        self.inputData = np.array(inputData)
        self.p = p
        
    def standarOpt(self):
        '''Standardized input data'''
        scaler = StandardScaler().fit(self.inputData) 
        self.inputData = scaler.transform(self.inputData)
        
    def buildModel(self):
        #The first column is the parent column, and the absolute difference with other columns is obtained
        momCol = self.inputData[:,0]
        sonCol = self.inputData[:,1:]

        for col in range(sonCol.shape[1]):
            sonCol[:,col] = abs(sonCol[:,col]-momCol)
        #Finding the minimum and maximum difference of two levels
        minMin = sonCol.min()
        maxMax = sonCol.max()
        #Calculation of correlation coefficient matrix
        cors = (minMin + self.p*maxMax)/(sonCol+self.p*maxMax)
        #Find the average comprehensive correlation degree
        meanCors = cors.mean(axis=0)
        self.result = {'cors':cors,'meanCors':meanCors}
    
    def get_cors(self):
        return self.result['cors']
    
    def get_meanCors(self):
        return self.result['meanCors']

if __name__ == "__main__":
    data=pd.read_csv('test_data.csv')
    print(data)
    model = GRA_Model(data)
    model.standarOpt()
    model.buildModel()
    print('----------------')
    meanCors=model.get_meanCors()
    x_column_namelst=data.columns.tolist()[1:]
    for i in range(len(meanCors)):
        print(x_column_namelst[i],'meanCors=',meanCors[i])        
