import numpy as np
import AdaBoost as ada


def main():
    data = file_get_contents('ressource/spambase.data')
    testSet = [data[0], data[30], data[1776], data[3805]]

    print(ada.error_rate(data, testSet, 3))
    print(ada.error_rate(data, testSet, 4))

def file_get_contents(filename):
    return np.genfromtxt(filename, delimiter=',')


if __name__ == "__main__":
    main()




