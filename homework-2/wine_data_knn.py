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


def show_statistics(k, dist):
    results = knn(k, dist)
    tmp = [[r[0], r[1]] for r in results]
    tp = tmp.count([WineData.HIGH, WineData.HIGH])
    fn = tmp.count([WineData.HIGH, WineData.LOW])
    fp = tmp.count([WineData.LOW, WineData.HIGH])
    tn = tmp.count([WineData.LOW, WineData.LOW])

    print('k = ' + str(k))
    print('-' * 30)

    print('Confusion Matrix')
    print(' ' * 8 + 'Predicted')
    print(' ' * 9 + '+' + ' ' * 5 + '-')
    print(' ' * 7 + '-' * 11)
    print(' ' * 5 + '+|' + '{0:4d} |{1:4d} |'.format(tp, fn))
    print('Actual|' + '-' * 11 + '|')
    print(' ' * 5 + '-|' + '{0:4d} |{1:4d} |'.format(fp, tn))
    print(' ' * 7 + '-' * 11)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * precision * recall / (precision + recall)

    print('Precision = ' + str(precision))
    print('Recall = ' + str(recall))
    print('F-measure = ' + str(f_measure))

    for r in results:
        if r[1] == WineData.LOW:
            r[2] = 1 - r[2]

    roc_points_x = [0]
    roc_points_y = [0]
    tpr = 0
    fpr = 0
    np = [r[0] for r in results].count(WineData.HIGH)

    for r in sorted(results, key=lambda x: x[-1], reverse=True):
        if r[0] == WineData.HIGH:
            tpr += 1 / np
        else:
            fpr += 1 / np

        roc_points_x.append(fpr)
        roc_points_y.append(tpr)

    roc_points_x.append(1)
    roc_points_y.append(1)

    pyplot.plot(roc_points_x, roc_points_y)
    pyplot.plot([0, 1], [0, 1])
    pyplot.show()

    print()
    print()


show_statistics(1, dist_euclid)
