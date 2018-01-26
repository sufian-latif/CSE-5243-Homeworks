from ted_data import TEDData, read_data
from pprint import pprint
from matplotlib import pyplot
import numpy as np

data = read_data('ted_main.csv')


def duration_histogram():
    interval = 5
    d = [d.duration / 60.0 for d in data]
    pyplot.xticks(range(0, int(max(d) + interval), interval))
    pyplot.hist(d, bins=50, edgecolor='black', linewidth=0.5)
    pyplot.axvline(np.mean(d), color='black')
    pyplot.show()


# duration_histogram()

def tag_groups():
    tag_frequency = dict()
    for d in data:
        for tag in d.tags:
            if tag in tag_frequency:
                tag_frequency[tag] += 1
            else:
                tag_frequency[tag] = 1
    # pprint([(k, tag_frequency[k]) for k in sorted(tag_frequency, key=tag_frequency.get, reverse=True)])

    group_size = 50
    tag_groups = dict()
    for tag in tag_frequency:
        gr = int(tag_frequency[tag] / group_size)
        if gr * group_size in tag_groups:
            tag_groups[gr * group_size].append(tag)
        else:
            tag_groups[gr * group_size] = [tag]


    pprint(tag_groups)

# tag_groups()

