import numpy as np


def extract_training(data):
    length = int(0.8 * len(data))
    return data[:length][:1813], data[:length][1813:]  # Training Dateien, getrennt je nach ob Spam ist


def extract_test(data):
    test = np.copy(data)
    np.random.shuffle(test)  # da die Dateien nicht zufällig gemischt sind, mischen wir hier selbst mal
    length = int(0.2 * len(test))  # dann nehmen wir 20% davon als unsere Test Dateien
    return test[:length]


def spam_class(data):
    return extract_training(data)[0]


def non_spam_class(data):
    return extract_training(data)[1]


def projection_linear(data):    # hier ist unseres Linear Koeffiziente W
    covarianceSpam = mycovariance(spam_class(data))
    covarianceNonSpam = mycovariance(non_spam_class(data))
    covarianceSum = get_it_invertible(np.add(covarianceSpam, covarianceNonSpam))
    inverseSum = np.linalg.inv(covarianceSum)

    middleSpam = diameter(spam_class(data))
    middleNonSpam = diameter(non_spam_class(data))
    middleDiffer = np.subtract(middleSpam, middleNonSpam)

    return np.dot(inverseSum, middleDiffer)


''' mit 'get_projection' Funktion können wir die projektion Funktion f(x) = Wt*x + W0 rechnen, 
 Wt ist die inverse Matrix von Projektion Koeffiziente, W0 ist eine Konstante, hier suchen wir so eine,
 dass mit der die Schätzung Richtigkeit vergrößert '''

def prediction_all(data, testsample):
    spam = spam_class(data)
    middleSpam = diameter(spam)
    nonSpam = non_spam_class(data)
    middleNonspam = diameter(nonSpam)

    w1 = np.matrix(projection_linear(data))
    w2 = np.transpose(w1)

    a = np.dot(middleSpam, w2)
    b = np.dot(middleNonspam, w2)
    w0 = -0.95*(a + b)         # die Konstante W0, laut Erfahrung irgendwelche eingesetzt :(

    if ((np.dot(testsample, w2) + w0) > 0):
        return 1
    else: return -1


def get_it_invertible(matrix):  # erzeuge einen invertierbaren Matrix
    a = 0.1
    id = np.identity(len(matrix))
    deter = np.linalg.det(matrix)

    while (deter == 0):    # wenn die Determinante 0 ist, gibt's keinen Inverse Matrix
        matrix = np.add(matrix, np.dot(a, id))
        deter = np.linalg.det(matrix)
    return matrix


def mycovariance(matrix):   # berechne die Kovarianzmatrix von unserer Trainingklasse
    middlevec = diameter(matrix)
    length = len(matrix)
    matrix1 = np.transpose(np.subtract(matrix, middlevec))
    matrix2 = np.transpose(matrix1)
    matrix = np.dot(matrix1, matrix2)
    return np.divide(matrix, length-1)


def diameter(images):    # hier wird die Mittelvektor vom Matrix berechnet
    result = np.zeros((len(images[0])), dtype=np.float)

    for pixel in range(0, len(images[0])): # kriegen an jedem Index von allen Sample die durchschnittliche Werte.
        sum = 0

        for image in range(0, len(images)):
            sum += images[image][pixel]

        result[pixel] = (sum / len(images))

    return result


def prediction(data):
    result = []
    for i in range(0, len(data)):
        tmp = prediction_all(data, data[i])
        result.append(tmp)
    return np.transpose(np.matrix(result))

