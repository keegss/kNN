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

# Associated img number in training set with img for use in making sub arrays
# for cross validation procedure
def genAssociatedArray(x_train):

    print('Generating associated trained set array...')

    tset = []
    for i in range(len(x_train)):
        tset.append((i, x_train[i]))

    print('Associated array created.')
    print('-----')
    return tset

# Generate all datasets for use in cross validation
def genFoldArrays(ax_train):

    print('Generating folds...')

    fold1_test = ax_train[0:6000]
    fold1_train = ax_train[6000:]

    fold2_test = ax_train[6000:12000]
    fold2_train = ax_train[0:6000]
    fold2_train.extend(ax_train[12000:])

    fold3_test = ax_train[12000:18000]
    fold3_train = ax_train[0:12000]
    fold3_train.extend(ax_train[18000:])

    fold4_test = ax_train[18000:24000]
    fold4_train = ax_train[0:18000]
    fold4_train.extend(ax_train[24000:])

    fold5_test = ax_train[24000:30000]
    fold5_train = ax_train[0:24000]
    fold5_train.extend(ax_train[30000:])

    fold6_test = ax_train[30000:36000]
    fold6_train = ax_train[0:30000]
    fold6_train.extend(ax_train[36000:])

    fold7_test = ax_train[36000:42000]
    fold7_train = ax_train[0:36000]
    fold7_train.extend(ax_train[42000:])

    fold8_test = ax_train[42000:48000]
    fold8_train = ax_train[0:42000]
    fold8_train.extend(ax_train[48000:])

    fold9_test = ax_train[48000:54000]
    fold9_train = ax_train[0:48000]
    fold9_train.extend(ax_train[54000:])

    fold10_test = ax_train[54000:60000]
    fold10_train = ax_train[:54000]

    fold1 = (fold1_test, fold1_train)
    fold2 = (fold2_test, fold2_train)
    fold3 = (fold3_test, fold3_train)
    fold4 = (fold4_test, fold4_train)
    fold5 = (fold5_test, fold5_train)
    fold6 = (fold6_test, fold6_train)
    fold7 = (fold7_test, fold7_train)
    fold8 = (fold8_test, fold8_train)
    fold9 = (fold9_test, fold9_train)
    fold10 = (fold10_test, fold10_train)

    print('Folds generated.')
    print('-----')

    return [fold1, fold2, fold3, fold3, fold5,
            fold6, fold7, fold8, fold9, fold10]

def euclideanDistance(a, b):
    dist = np.sqrt(np.sum((a-b)**2))
    return dist

def getDistances(trainingSet, imgTest, y_train):
    distances = []
    for i in range(len(trainingSet)):
        dis = euclideanDistance(imgTest, trainingSet[i][1])
        distances.append((y_train[trainingSet[i][0]], dis))

    distances.sort(key=operator.itemgetter(1))

    ret = []
    for i in range(0, 10):
        ret.append((distances[i][0], distances[i][1]))

    return ret

def generateAllDistances(fold, data, y_train):
    test = data[0]
    train = data[1]

    allDis = []

    print('Generating distances array for ' + fold + '...')
    t0 = time.time()
    for i in range(len(test)):
        allDis.append((y_train[test[i][0]],  getDistances(train, test[i][1], y_train)))
    t1 = time.time()

    print('Distances array generated. Time taken: ')
    print(t1-t0)
    print('-----')

    return allDis

def kNN(distances, k):

    neighbors = []
    for i in range(0, k):
        neighbors.append(distances[i])

    prediction = Counter(neighbors).most_common(1)[0]

    return prediction[0][0]

def find_k_accuracy(distances):

    howGoodK = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for j in range(len(distances)):     # loop through all distance arrays
        actual = distances[j][0]
        for p in range(1, 11):          # loop through ks
            prediction = kNN(distances[j][1], p)
            if prediction == actual:
                howGoodK[p - 1] += 1

    return howGoodK


def cross_validation(x_train, y_train, x_test, y_test):

    ax_train = genAssociatedArray(x_train)
    folds = genFoldArrays(ax_train)

	# grab distances array for each fold...
    print('Generating distances arrays for each fold.')
    print('This is going to take a very long time...')

    print('-----')
    t0 = time.time()
    fold1_distances = generateAllDistances('Fold 1', folds[0], y_train)

    pickling
    with open("fold1.txt", "wb") as fp:
       pickle.dump(fold1_distances, fp)


    fold2_distances = generateAllDistances('Fold 2', folds[1], y_train)

    pickling
    with open("fold2.txt", "wb") as fp:
       pickle.dump(fold2_distances, fp)

    fold3_distances = generateAllDistances('Fold 3', folds[2], y_train)

    pickling
    with open("fold3.txt", "wb") as fp:
       pickle.dump(fold3_distances, fp)

    fold4_distances = generateAllDistances('Fold 4', folds[3], y_train)

    pickling
    with open("fold4.txt", "wb") as fp:
       pickle.dump(fold4_distances, fp)

    fold5_distances = generateAllDistances('Fold 5', folds[4], y_train)

    pickling
    with open("fold5.txt", "wb") as fp:
       pickle.dump(fold5_distances, fp)

    fold6_distances = generateAllDistances('Fold 6', folds[5], y_train)

    pickling
    with open("fold6.txt", "wb") as fp:
       pickle.dump(fold6_distances, fp)

    fold7_distances = generateAllDistances('Fold 7', folds[6], y_train)

    pickling
    with open("fold7.txt", "wb") as fp:
       pickle.dump(fold7_distances, fp)

    fold8_distances = generateAllDistances('Fold 8', folds[7], y_train)

    pickling
    with open("fold8.txt", "wb") as fp:
       pickle.dump(fold8_distances, fp)

    fold9_distances = generateAllDistances('Fold 9', folds[8], y_train)

    pickling
    with open("fold9.txt", "wb") as fp:
       pickle.dump(fold9_distances, fp)

    fold10_distances = generateAllDistances('Fold 10', folds[9], y_train)

    pickling
    with open("fold10.txt", "wb") as fp:
       pickle.dump(fold10_distances, fp)

    t1 = time.time()
    print('-----')

    print('All distances calcualted. Total time taken: ')
    print(t1-t0)

    # load results
    # with open("fold1.txt", "rb") as fp:
    #     fold1_distances = pickle.load(fp)
    # with open("fold2.txt", "rb") as fp:
    #     fold2_distances = pickle.load(fp)
    # with open("fold3.txt", "rb") as fp:
    #     fold3_distances = pickle.load(fp)
    # with open("fold4.txt", "rb") as fp:
    #     fold4_distances = pickle.load(fp)
    # with open("fold5.txt", "rb") as fp:
    #     fold5_distances = pickle.load(fp)
    # with open("fold6.txt", "rb") as fp:
    #     fold6_distances = pickle.load(fp)
    # with open("fold7.txt", "rb") as fp:
    #     fold7_distances = pickle.load(fp)
    # with open("fold8.txt", "rb") as fp:
    #     fold8_distances = pickle.load(fp)
    # with open("fold9.txt", "rb") as fp:
    #     fold9_distances = pickle.load(fp)
    # with open("fold10.txt", "rb") as fp:
    #     fold10_distances = pickle.load(fp)

    k_fold1 = find_k_accuracy(fold1_distances)
    k_fold2 = find_k_accuracy(fold2_distances)
    k_fold3 = find_k_accuracy(fold3_distances)
    k_fold4 = find_k_accuracy(fold4_distances)
    k_fold5 = find_k_accuracy(fold5_distances)
    k_fold6 = find_k_accuracy(fold6_distances)
    k_fold7 = find_k_accuracy(fold7_distances)
    k_fold8 = find_k_accuracy(fold8_distances)
    k_fold9 = find_k_accuracy(fold9_distances)
    k_fold10 = find_k_accuracy(fold10_distances)

    kNum = []

    kNum.append(k_fold1)
    kNum.append(k_fold2)
    kNum.append(k_fold3)
    kNum.append(k_fold4)
    kNum.append(k_fold5)
    kNum.append(k_fold6)
    kNum.append(k_fold7)
    kNum.append(k_fold8)
    kNum.append(k_fold9)
    kNum.append(k_fold10)

    kAccr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(0, 10):
        for j in range(0, 10):
            kAccr[i] += kNum[j][i]

    print('Total correct predictions for each k throughout all 10 folds.')
    print(kAccr)

    total = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(0, 10):
        total[i] += kAccr[i] / (6000 * 10)

    for i in range(0, 10):
        print("K = %d ; Accuracy = %.5f" % (i + 1, total[i]))



def main():
    x_train, y_train, x_test, y_test = load_data()

    x_train.astype(float)
    x_test.astype(float)
    y_train.astype(float)
    y_test.astype(float)

    cross_validation(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()
