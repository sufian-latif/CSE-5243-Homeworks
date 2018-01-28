import ted_data
from pprint import pprint
from matplotlib import pyplot, dates
import numpy as np

data = ted_data.read_data('ted_main.csv')


def duration_histogram():
    interval = 1  # 5-minutes interval
    d = [d.duration for d in data]
    bins = range(0, int(max(d)), interval)
    pyplot.grid(axis='y', zorder=0)
    pyplot.xticks(bins[::5])
    pyplot.hist(d, bins=bins, edgecolor='black', linewidth=0.5, zorder=2)
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
    pyplot.grid(axis='y', zorder=0)
    pyplot.xticks(range(len(top_tags)), top_tags, rotation=90)
    pyplot.bar(range(len(top_tags)), [tag_frequency[t] for t in top_tags], zorder=2)
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


def events_histogram():
    events_group = dict()
    for d in data:
        events_group[d.event] = events_group.get(d.event, 0) + 1

    top_events = sorted(events_group, key=events_group.get, reverse=True)[:20]
    # pprint(top_events)
    pyplot.xticks(range(len(top_events)), top_events, rotation=90)
    pyplot.bar(range(len(top_events)), [events_group[e] for e in top_events])
    pyplot.show()


# events_histogram()


def frequency_over_years():
    filmed = dict()
    published = dict()
    for d in data:
        filmed[d.film_date.year] = filmed.get(d.film_date.year, 0) + 1
        published[d.published_date.year] = published.get(d.published_date.year, 0) + 1

    pyplot.grid()
    pyplot.xticks(range(min(filmed.keys()), max(filmed.keys()) + 1), rotation=90)
    pyplot.plot([k for k in sorted(filmed.keys())],
                [filmed[k] for k in sorted(filmed.keys())], 'o-')
    pyplot.plot([k for k in sorted(published.keys())],
                [published[k] for k in sorted(published.keys())], 'o-')
    pyplot.show()


# frequency_over_years()


def time_to_publish():
    cut = 200
    x_interval = 250
    y_interval = 2

    d_film = sorted(data, key=lambda d: d.film_date)[:]
    y_film = [dates.date2num(d.film_date) / 365 for d in d_film]
    y_publish = [dates.date2num(d.published_date) / 365 for d in d_film]
    tstamps = set([int(y) for y in y_film + y_publish])

    pyplot.grid(axis='y')
    pyplot.xticks(range(0, len(d_film) + x_interval, x_interval))
    pyplot.yticks(range(min(tstamps), max(tstamps) + y_interval, y_interval))
    pyplot.plot(range(len(d_film)), y_film, '-', linewidth=0.75)
    pyplot.plot(range(len(d_film)), y_publish, '-', linewidth=0.75)
    pyplot.fill_between(range(len(d_film)), y_film, y_publish, facecolor='xkcd:ice blue')
    pyplot.axvline(cut, color='black')
    pyplot.ticklabel_format()
    pyplot.show()


# time_to_publish()

# print([d.published_date > d.film_date for d in data].count(False))


def language_histogram():
    interval = 1
    d = [d.languages for d in data]
    bins = range(0, int(max(d)), interval)
    pyplot.grid(axis='y', zorder=0)
    pyplot.xticks(bins[::5])
    pyplot.hist(d, bins=bins, edgecolor='black', linewidth=0.5, zorder=2)
    pyplot.axvline(np.mean(d), color='black')
    pyplot.show()


language_histogram()