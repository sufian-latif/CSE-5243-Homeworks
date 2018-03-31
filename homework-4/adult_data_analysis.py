import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pprint import pprint

columns = ["age", "workclass", "fnlwgt", "education", "education-Num", "martial-status", "occupation",
           "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "country", "class"]

data = pd.read_csv("adult.data", names=columns, sep=r'\s*,\s*', engine='python', na_values=['?']).dropna()
test_data = pd.read_csv("adult.test", names=columns, sep=r'\s*,\s*', engine='python', na_values=['?']).dropna()


def histogram():
    fig = plt.figure()
    cols = 3
    rows = ceil(float(data.shape[1]) / cols)
    for i, column in enumerate(data.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column, bbox=dict(facecolor='white', edgecolor='black'))
        if data.dtypes[column] == np.object:
            print(data[column].value_counts())
            data[column].value_counts().plot(kind='bar', axes=ax)
            plt.xticks(fontsize=8)
        else:
            data[column].hist(axes=ax, grid=False)
            # plt.xticks(rotation='vertical')
    # plt.subplots_adjust(hspace=1.5, wspace=0.5)
    plt.show()


# histogram()


def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


enc_data, _ = number_encode_features(data)
enc_test_data, _ = number_encode_features(test_data)

sns.set(font_scale=1)
sns.heatmap(enc_data.corr(), square=True, cmap='coolwarm', annot=True, fmt='.2f')
plt.show()


def get_result(x_train, x_test, y_train, y_test, classifier):
    # print(classifier)
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    cm = metrics.confusion_matrix(y_test, pred)
    probs = classifier.predict_proba(x_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, probs[:, 1])
    stats = {'Confusion matrix': cm,
             'Accuracy': metrics.accuracy_score(y_test, pred),
             'Precision': metrics.precision_score(y_test, pred),
             'Recall': metrics.recall_score(y_test, pred),
             'F-measure': metrics.f1_score(y_test, pred),
             'AUC': metrics.auc(fpr, tpr),
             'ROC data': [fpr, tpr]}

    return stats


def logistic_regression():
    train = enc_data
    test = enc_test_data

    x_train, x_test, y_train, y_test = train.drop('class', axis=1), test.drop('class', axis=1), train['class'],\
                                       test['class']
    return get_result(x_train, x_test, y_train, y_test, LogisticRegression(penalty='l1'))


def svm():
    scaler = StandardScaler()
    scaler.fit(enc_data.drop('class', axis=1))
    train = scaler.transform(enc_data.drop('class', axis=1))
    test = scaler.transform(enc_test_data.drop('class', axis=1))

    x_train, x_test, y_train, y_test = train, test, enc_data['class'], enc_test_data['class']
    return get_result(x_train, x_test, y_train, y_test, SVC(probability=True))


def random_forest():
    train = enc_data
    test = enc_test_data

    x_train, x_test, y_train, y_test = train.drop('class', axis=1), test.drop('class', axis=1), train['class'],\
                                       test['class']
    return get_result(x_train, x_test, y_train, y_test, RandomForestClassifier(n_estimators=100, criterion='entropy'))


def neural_net():
    scaler = StandardScaler()
    scaler.fit(enc_data.drop('class', axis=1))
    train = scaler.transform(enc_data.drop('class', axis=1))
    test = scaler.transform(enc_test_data.drop('class', axis=1))

    x_train, x_test, y_train, y_test = train, test, enc_data['class'], enc_test_data['class']
    return get_result(x_train, x_test, y_train, y_test, MLPClassifier(hidden_layer_sizes=(200, )))


def decision_tree():
    train = enc_data
    test = enc_test_data

    x_train, x_test, y_train, y_test = train.drop('class', axis=1), test.drop('class', axis=1), train['class'],\
                                       test['class']
    return get_result(x_train, x_test, y_train, y_test, DecisionTreeClassifier(criterion='entropy'))


results = {
    # 'Decision tree': decision_tree(),
    # 'Neural network': neural_net(),
    # 'Support vector machine': svm(),
    # 'Random forest': random_forest(),
    # 'Logistic regression': logistic_regression()
}


def plot_roc():
    plt.plot([0, 1], [0, 1], lw=0.75, linestyle='--', color='gray')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    for name in results:
        plt.plot(results[name]['ROC data'][0], results[name]['ROC data'][1],
                 label=name + f' (area = {results[name]["AUC"]:.3})', lw=0.75)

    plt.legend()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()


def plot_radar():
    labels = ['Accuracy', 'Precision', 'Recall', 'F-measure', 'AUC']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    ax = plt.subplot(111, projection='polar')
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_theta_offset(angles[1])
    ax.set_ylim(0, 1)

    for name in results:
        values = [results[name][k] for k in labels + [labels[0]]]
        ax.plot(angles, values, 'o-', lw=1, label=name)
        ax.fill(angles, values, alpha=0.05)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.show()


# pprint(results)
# plot_roc()
# plot_radar()
