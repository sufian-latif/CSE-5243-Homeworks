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
    tag_views = dict()
    for d in data:
        for tag in d.tags:
            if tag == 'TEDx':
                continue
            tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
            tag_views[tag] = tag_views.get(tag, 0) + d.views
    # pprint([(k, tag_frequency[k]) for k in sorted(tag_frequency, key=tag_frequency.get, reverse=True)])
    top_tags = sorted(tag_frequency, key=tag_frequency.get, reverse=True)[:20]
    pyplot.grid(axis='y', zorder=0)
    pyplot.xticks(range(len(top_tags)), top_tags, rotation=90)
    pyplot.bar(range(len(top_tags)), [tag_frequency[t] for t in top_tags], zorder=2)
    pyplot.show()

    top_viewd_tags = sorted(tag_views, key=tag_views.get, reverse=True)[:20]
    pyplot.grid(axis='y', zorder=0)
    pyplot.xticks(range(len(top_viewd_tags)), top_viewd_tags, rotation=90)
    pyplot.bar(range(len(top_viewd_tags)), [tag_views[t] for t in top_viewd_tags], zorder=2)
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


# language_histogram()


def language_view_scatter():
    d_lang = [d.languages for d in data]
    d_view = [d.views for d in data]

    pyplot.scatter(d_lang, d_view)
    pyplot.show()


# language_view_scatter()


def duration_view_scatter():
    d_dur = [d.duration for d in data]
    d_view = [d.views for d in data]

    pyplot.scatter(d_dur, d_view)
    pyplot.show()


# duration_view_scatter()

def occupation_group():
    occ_frequency = dict()
    occ_views = dict()
    for d in data:
        for occ in d.speaker_occupation:
            occ_frequency[occ] = occ_frequency.get(occ, 0) + 1
            occ_views[occ] = occ_views.get(occ, 0) + d.views

    print(len(occ_frequency))
    # pprint([(k, tag_frequency[k]) for k in sorted(tag_frequency, key=tag_frequency.get, reverse=True)])
    top_tags = sorted(occ_frequency, key=occ_frequency.get, reverse=True)[:20]
    pyplot.grid(axis='y', zorder=0)
    pyplot.xticks(range(len(top_tags)), top_tags, rotation=90)
    pyplot.bar(range(len(top_tags)), [occ_frequency[t] for t in top_tags], zorder=2)
    pyplot.show()

    top_viewed_tags = sorted(occ_views, key=occ_views.get, reverse=True)[:20]
    pyplot.grid(axis='y', zorder=0)
    pyplot.xticks(range(len(top_viewed_tags)), top_viewed_tags, rotation=90)
    pyplot.bar(range(len(top_viewed_tags)), [occ_views[t] for t in top_viewed_tags], zorder=2)
    pyplot.show()


# occupation_group()

def tag_trend():
    years = range(2006, 2017)
    tag_frequency = dict()
    tag_views = dict()
    for d in data:
        if d.film_date.year > 2005:
            for tag in d.tags:
                if tag == 'TEDx':
                    continue
                tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
                tag_views[tag] = tag_views.get(tag, 0) + d.views
    top_tags = sorted(tag_frequency, key=tag_frequency.get, reverse=True)[:10]

    pyplot.xticks(range(len(years)), years)
    pyplot.grid(axis='x')

    for tag in top_tags:
        counts = [len([d for d in data if d.film_date.year == year and tag in d.tags]) for year in years]
        pyplot.plot(counts, label=tag)

    pyplot.legend()
    pyplot.show()


# tag_trend()

