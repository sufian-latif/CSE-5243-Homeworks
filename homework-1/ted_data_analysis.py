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