import numpy as np
import pickle
import math
import operator
import time
import itertools
import copy
from collections import Counter

def load_data():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return x_train, y_train, x_test, y_test

def make_windows():
    n = 30
    m = 30
    window = [1] * n
    for i in range(n):
        window[i] = [1] * m

    # make a copy of the 30x30 matrix of ones
    window1 = copy.deepcopy(window)
    window2 = copy.deepcopy(window)
    window3 = copy.deepcopy(window)

    window1[28] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window1[29] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    window2[0]  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window2[29] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window3[0]  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window3[1]  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(0, 30):
        window1[i][29] = 0
        window1[i][28] = 0
        window2[i][29] = 0
        window2[i][28] = 0
        window3[i][29] = 0
        window3[i][28] = 0

    window4 = copy.deepcopy(window)
    window5 = copy.deepcopy(window)
    window6 = copy.deepcopy(window)

    window4[28] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window4[29] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window5[0]  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window5[29] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window6[0]  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window6[1]  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(0, 30):
        window4[i][29] = 0
        window4[i][0] = 0
        window5[i][29] = 0
        window5[i][0] = 0
        window6[i][29] = 0
        window6[i][0] = 0

    window7 = copy.deepcopy(window)
    window8 = copy.deepcopy(window)
    window9 = copy.deepcopy(window)

    window7[28] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window7[29] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window8[0]  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window8[29] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window9[0]  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    window9[1]  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(0, 30):
        window7[i][0] = 0
        window7[i][1] = 0
        window8[i][0] = 0
        window8[i][1] = 0
        window9[i][0] = 0
        window9[i][1] = 0

    return [window1, window2, window3, window4, window5, window6, window7, window8, window9]

def euclideanDistance(a, b):
    dist = np.sqrt(np.sum((a-b)**2))
    return dist

def getWindowDistances(trainingSet, imgTest, y_train, windows):
    test_padded = np.pad(imgTest, (1,1), 'constant')

    distances = []

    for i in range(len(trainingSet)):
        window_dis = []
        train_padded = np.pad(trainingSet[i], (1,1), 'constant')
        for j in range(len(windows)):
            snip = train_padded * windows[j]
            dis = euclideanDistance(test_padded, snip)
            window_dis.append((y_train[i], dis))

        window_dis.sort()
        distances.append(window_dis[0])

    distances.sort(key=operator.itemgetter(1))
    ret = []
    for i in range(0, 10):
        ret.append((distances[i][0], distances[i][1]))

    return ret

def genDistancesWindow(x_test, x_train, y_train):

    test = x_test
    train = x_train

    windows = make_windows()

    allDis = []

    print('Generating distances array for sliding window...')
    t0 = time.time()
    for i in range(0, 100): #len(test)):
        allDis.append((i, getWindowDistances(train, test[i], y_train, windows)))
    t1 = time.time()

    print('Distances array generated. Time taken ')
    print(t1-t0)
    print('-----')

    return allDis

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

def sliding_window_knn(x_test, x_train, y_train, y_test):

    windows = make_windows()
    distances = genDistancesWindow(x_test, x_train, y_train)

    #with open("window_distances.txt", "wb") as fp:
    #    pickle.dump(distances, fp)

	#with open("window_distances.txt", "rb") as fp:
    #    pickle.dump(distances, fp)

    correct_predictions, final_matrix = find_accuracy(distances, 1)

    return correct_predictions, final_matrix


def main():
    x_train, y_train, x_test, y_test = load_data()
    correct_predictions, confusion_matrix = sliding_window_knn(x_test, x_train, y_train, y_test)

    print('Total correct predictions for test images')
    print(correct_predictions)
    print('Confusion matrix')
    print(confusion_matrix)

    print("Final accuracy")
    print("K = 1 ; Accuracy = %.5f" % (correct_predictions / 60000))

if __name__ == "__main__":
    main()
