from pprint import pprint
from random import shuffle
from wine_data import read_data, WineData
from matplotlib import pyplot
import numpy as np

wine_data = read_data('winequality-red.csv')
shuffle(wine_data)
cut = len(wine_data) * 3 // 4
train_data, test_data = wine_data[:cut], wine_data[cut:]


def dist_euclid(d1, d2):
    return sum([(d1.data[i] - d2.data[i]) ** 2 for i in range(len(d1.data))]) ** 0.5


def dist_manhattan(d1, d2):
    return sum([abs(d1.data[i] - d2.data[i]) for i in range(len(d1.data))])


def knn(k, dist):
    results = []
    for td in test_data:
        nn = sorted([(dist(d, td), d.quality) for d in train_data])[:k]
        prob = dict((q, [x[1] for x in nn].count(q) / len(nn)) for q in (WineData.HIGH, WineData.LOW))
        pred = WineData.HIGH if prob[WineData.HIGH] > prob[WineData.LOW] else WineData.LOW
        results.append([td.quality, pred, prob[pred]])

    return results

# pprint(knn(3, dist_euclid))
