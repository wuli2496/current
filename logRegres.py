# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:11:26 2021

@author: wl
"""

import math
from numpy import mat
from numpy import shape
from numpy import ones

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
        
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + math.exp(-inX))

def gradAscent(dataMatIn, classLables):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLables).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
        
    return weights