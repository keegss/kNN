import numpy as np
import pickle
import math
import operator
import time
import itertools
from collections import Counter

def load_data():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return x_train, y_train, x_test, y_test

def euclideanDistance(a, b):
    dist = np.sqrt(np.sum((a-b)**2))
    return dist

def getDistancesArray(trainingSet, imgTest, y_train):
    distances = []
    for i in range(len(trainingSet)):
        dis = euclideanDistance(imgTest, trainingSet[i])
        distances.append((y_train[i], dis))

    distances.sort(key=operator.itemgetter(1))

    ret = []
    for i in range(0, 10):
        ret.append((distances[i][0], distances[i][1]))

    return ret

def generateAllDistances(x_train, y_train, x_test, y_test):

    distances = []
    for i in range(0, len(x_test)):
        distances.append((y_test[i], getDistancesArray(x_train, x_test[i], y_train)))

    with open("test.txt", "wb") as fp:
       pickle.dump(distances, fp)

    return distances

def majorityVote(distances, k):

    neighbors = []
    for i in range(0, k):
        neighbors.append(distances[i])

    prediction = Counter(neighbors).most_common(1)[0]

    return prediction[0][0]

def find_accuracy(distances, k):
    correct_predictions = 0
    confusion_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]

    for j in range(len(distances)):
        actual = distances[j][0]
        prediction = majorityVote(distances[j][1], k)
        if prediction == actual:
            correct_predictions += 1
        confusion_matrix[prediction][actual] += 1

    return correct_predictions, confusion_matrix

def kNN(x_train, y_train, x_test, y_test):

    distances = generateAllDistances(x_train, y_train, x_test, y_test)

    #with open("test.txt", "rb") as fp:
    #    distances = pickle.load(fp)

    correct_predictions, final_matrix = find_accuracy(distances, 2)

    return correct_predictions, final_matrix

def main():

    x_train, y_train, x_test, y_test = load_data()

    x_train.astype(float)
    x_test.astype(float)
    y_train.astype(float)
    y_test.astype(float)

    correct_predictions, confusion_matrix = kNN(x_train, y_train, x_test, y_test)

    print('Total correct predictions for test images')
    print(correct_predictions)
    print('Confusion matrix')
    for i in range(len(confusion_matrix)):
	    print(confusion_matrix[i])

    print("Final accuracy")
    print("K = 2 ; Accuracy = %.5f" % (correct_predictions / 60000))


if __name__ == "__main__":
    main()
