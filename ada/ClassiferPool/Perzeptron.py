import numpy as np


'''nehmen wir 20% zufälligen Dateien als Testdatein'''
def get_testdata(data):
    length = int(0.2 * len(data))
    return data[:length]


def get_trainingdata(data):
    length = int(1 * len(data))
    return data[:length]


def initial_weight(x):

    length = len(x)-1
    y = x[length]

    w = np.random.randint(10, size=(length, 1))
    tmp = np.dot(y, np.transpose(w))

    return np.dot(tmp, np.array(x[:-1]))


def get_weight_interativ(training):
    length = len(training)
    column = np.ones((len(training), 1))
    training = np.concatenate((column, training), axis=1)
    w = np.transpose(np.random.randint(10, size=(len(training[0] - 1), 1)))
    b = 0
    # mit nicht linear separierbaren Datensätzen benuzten wir iterative Aufrufe, hier mit iterative Begrenzung 5000
    for i in range(0, 50000):
        i = np.random.randint(0, length)
        x = training[i]
        if (x[-1] == 1.0 and (np.dot(w, x) + b) < 0):

            w = update(w, b, x)[0]
            b = update(w, b, x)[1]
        if (x[-1] == -1.0 and (np.dot(w, x) + b) > 0):

            x = np.dot(-1, x)

            w = update(w, b, x)[0]
            b = update(w, b, x)[1]
        if ((np.dot(w, x) + b) == 0):
            pass
    return w, b


def all_checked(w, b, training):
    for i in range(0, len(training)):
        x = training[i]
        if (x[-1] == 1.0):
            if ((np.dot(w, x) + b) < 0):
                return False
        elif (np.dot(w, x) >= 0):
            return False
    return True

''' mit update Funktion wird das Gewicht Vektor und bias immer aktualisiert, w = w + ηy(i)x(i), b = b + ηy(i)
0.5 entspricht hier die learning rate, welches im Interval (0, 1) liegt'''

def update(w, b ,x):
    y = x[-1]
    w = np.add(w, 0.8*(np.dot(y, x)))
    b = b + 0.8*y
    return w, b


def prediction(data):
    weight = get_weight_interativ(data)[0]
    weight = np.transpose(np.transpose(weight)[:-1])
    bias = get_weight_interativ(data)[1]
    result = []
    for i in range(0, len(data)):
        test = np.transpose(np.matrix(data[i]))
        tmp = np.dot(weight, test) + bias
        if (tmp >= 0):
            result.append(-1)
        else: result.append(1)
    return np.transpose(np.matrix(result))

def prediction_all(data, test):
    weight = get_weight_interativ(data)[0]
    weight = np.transpose(np.transpose(weight)[:-1])
    bias = get_weight_interativ(data)[1]
    test = np.transpose(np.matrix(test))
    if ((np.dot(weight, test) + bias) >= 0):
        return -1
    else: return 1