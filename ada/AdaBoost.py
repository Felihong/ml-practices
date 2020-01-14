import numpy as np
''' wir packen 3 Klassifizieren Algorithmen in unsere Klassenpool '''
import ClassiferPool.FischerDiskriminante as fischer
import ClassiferPool.Perzeptron as perzeptron
import ClassiferPool.KNN as knn


def initial_weight_data(data):
    return np.ones((len(data), 1))


def initial_weight_class(MaxInter):
    return np.zeros((MaxInter, 1))


def sigmoid(x):
    tmp = 1.0 + np.exp(-x)
    result = 1.0 / tmp
    return result

''' hier berechnen wir das exponential Error von den Gewicht '''
def cal_error_weight(data, dataWeightVector, resultVector):
    sum = 0
    errorIndex = []
    for i in range(0, len(data)):
        if (data[i][-1] != resultVector[i]):
            sum += dataWeightVector[i]
            errorIndex.append(i)
    return sum, errorIndex

''' hier wählen wir die Klasse aus dem Klassenpool, die das exponentiale Error minimiert,
 gleichzeitig merken wir die Hit Dateien und Miss Dateien, damit wir den Datensatz später passend 
 aktualisieren '''
def get_next_class(data, dataWeight):
    arrayClass = [fischer, knn, perzeptron]
    arrayError = []
    errorIndex = []
    #print('ja')
    result1 = fischer.prediction(data)
    #print('hello')
    arrayError.append(cal_error_weight(data, dataWeight, result1)[0])
    errorIndex.append(cal_error_weight(data, dataWeight, result1)[1])

    result2 = knn.prediction(data)
    #print('hi')
    arrayError.append(cal_error_weight(data, dataWeight, result2)[0])
    errorIndex.append(cal_error_weight(data, dataWeight, result2)[1])

    result3 = perzeptron.prediction(data)
    #print('hey')
    arrayError.append(cal_error_weight(data, dataWeight, result3)[0])
    errorIndex.append(cal_error_weight(data, dataWeight, result3)[1])

    index = np.argmin(arrayError)
    error = np.amin(arrayError)

    return arrayClass[index], error, errorIndex


def adaBoosting(data, MaxInter):
    classWeight = initial_weight_class(MaxInter)
    dataWeight = initial_weight_data(data)
    classPool = []

    for i in range(0, MaxInter):
        result = get_next_class(data, dataWeight)
        #print(result[1])
        classPool.append(result[0]) # füge neue Klasse hinzu

        e = result[1] / np.sum(dataWeight)
        right = 0.5 * (np.log((1-e)/e))
        classWeight[i] = right  # aktualisiere das Gewicht von der neuen Klasse

        errorIndex = result[2]
        update_data_weight(dataWeight, errorIndex, right) # aktualisiere das Gewicht des Datensätzes

    return classPool, classWeight # wir bekommen unsere Committee und deren "speak right"

''' aktualisieren den Datensatz, je nach ob es richtig geschätzt wird'''
def update_data_weight(dataWeight, errorIndex, right):
    j = 0
    for i in range(0, len(dataWeight)):
        if (i != errorIndex[j]):
            dataWeight[i] = dataWeight[i] * np.exp(right)
        else:
            dataWeight[i] = dataWeight[i] * np.exp(0-right)
            j += 1
    return dataWeight


def prediction(data, testsample, MaxInter):
    adaBoost = adaBoosting(data, MaxInter)
    classes = adaBoost[0]
    weights = adaBoost[1]
    resultArray = []

    for i in range(0, len(classes)):
        result = classes[i].prediction_all(data, testsample)
        resultArray.append(result)

    C = np.dot(np.transpose(weights), np.transpose(np.matrix(resultArray)))

    if (sigmoid(C) >= 0.5):
        return 1
    else: return 0


def error_rate(data, test, MaxInter):
    error = 0

    for i  in range(0, len(test)):
        result = prediction(data, test[i], MaxInter)
        print(result)
        if (result != test[i][-1]):
            error += 1
    print('error rate mit', MaxInter, 'Iterationen ist', error / len(test))




