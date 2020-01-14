import math
import numpy as np


def prediction_all(image, array):
    inf = math.inf
    acc = 0   # initiale index

    for i in range(0, len(array)):
        d = distance(image, array[i])

        if inf > d:
            inf = d
            acc = i    # suchen wir die aktuell kleinste Anstand mit ihre Index

    return array[acc][0]
#d = distance(array[i], image)

def smallassist(image, array):
    inf = math.inf
    acc = 0

    for i in range(0, len(array)):
        d = distance(image, array[i])
        if inf > d:
            inf = d
            acc = i
    return acc



def secsmall(image, array):
    smallindex = smallassist(image,array)
    inf = math.inf
    acc = 0

    for i in range(0,len(array)):
        if i == smallindex:
            i += 1
            d = distance(image, array[i])
            if inf > d:
                inf = d
                acc = i
    return array[acc][0]

def secsmallassist(image,array):
    smallindex = smallassist(image, array)
    inf = math.inf
    acc = 0

    for i in range(0, len(array)):
        if i == smallindex:
            i += 1
            d = distance(image, array[i])
            if inf > d:
                inf = d
                acc = i
    return acc


def dritsmall(image,array):
    smallindex =smallassist(image,array)
    secsmallindex =secsmallassist(image,array)
    inf=math.inf
    acc=0
    for i in range(0,len(array)):
        if i == smallindex:
            i += 1
        elif i==secsmallindex:
            i=i+1

        d = distance(image, array[i])
        if inf > d:
                inf = d
                acc = i
    return array[acc][0]


def distance(array1, array2):
    length = len(array1)

    if length != len(array2):
        print('error1')

    dist = 0

    for i in range(1, length):
        dist += math.pow((array1[i] - array2[i]), 2)
    else:
        dist = math.sqrt(dist)   # jede Matrix ist 16*16, mit 256+1 Index, die Abstand wird gerechnet

    return dist

def prediction(data):
    result = []
    for i in range(0, len(data)):
        tmp = int(prediction_all(data[i], data))
        #tmp = int(prediction_all(data, data[i]))
        if (tmp == 0):
            tmp = -1
        result.append(tmp)
    return np.transpose(np.matrix(result))





