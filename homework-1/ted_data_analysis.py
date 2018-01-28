from ted_data import TEDData, read_data
from pprint import pprint
from matplotlib import pyplot
import numpy as np

data = read_data('ted_main.csv')


def duration_histogram():
    interval = 1  # 5-minutes interval
    d = [d.duration for d in data]
    bins = range(0, int(max(d) + interval), interval)
    pyplot.xticks(bins[::5])
    pyplot.yscale('log')
    pyplot.hist(d, bins=bins, edgecolor='black', linewidth=0.5)
    pyplot.axvline(np.mean(d), color='black')
    pyplot.show()


# duration_histogram()


def tag_groups():
    tag_frequency = dict()
    for d in data:
        for tag in d.tags:
            tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
    # pprint([(k, tag_frequency[k]) for k in sorted(tag_frequency, key=tag_frequency.get, reverse=True)])
    top_tags = sorted(tag_frequency, key=tag_frequency.get, reverse=True)[:20]
    # pprint(top_tags)
    pyplot.xticks(range(len(top_tags)), top_tags, rotation=90)
    pyplot.bar(range(len(top_tags)), [tag_frequency[t] for t in top_tags])
    pyplot.show()


# tag_groups()


def views_histogram():
    interval = 500
    d = [d.views / 1000.0 for d in data]  # k-views
    bins = range(0, int(max(d) + interval), interval)
    pyplot.xticks(bins[::10])
    pyplot.yscale('log')
    pyplot.hist(d, bins=bins, edgecolor='black', linewidth=0.5)
    pyplot.axvline(np.mean(d), color='black')
    pyplot.show()


# views_histogram()

