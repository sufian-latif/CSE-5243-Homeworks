import ted_data
from pprint import pprint
from matplotlib import pyplot, dates, cm
import numpy as np
import networkx as nx
from functools import reduce

ted_data = ted_data.read_data('ted_main.csv')
tedx_data = [d for d in ted_data if 'TEDx' in d.tags]


def duration_histogram():
    interval = 1
    d = [d.duration for d in ted_data]
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
    for d in ted_data:
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
    d = [d.views / 1000.0 for d in ted_data]  # k-views
    bins = range(0, int(max(d) + interval), interval)
    pyplot.xticks(bins[::10])
    pyplot.yscale('log')
    pyplot.hist(d, bins=bins, edgecolor='black', linewidth=0.5)
    pyplot.axvline(np.mean(d), color='black')
    pyplot.show()


# views_histogram()


def events_histogram():
    events_group = dict()
    for d in ted_data:
        events_group[d.event] = events_group.get(d.event, 0) + 1

    top_events = sorted(events_group, key=events_group.get, reverse=True)[:20]
    # pprint(top_events)
    pyplot.xticks(range(len(top_events)), top_events, rotation=90)
    pyplot.bar(range(len(top_events)), [events_group[e] for e in top_events])
    pyplot.show()


# events_histogram()


def frequency_over_years():
    filmed = dict()
    filmed_tedx = dict()
    published = dict()
    for d in ted_data:
        if d.film_date.year < 2001:
            continue
        if 'TEDx' in d.tags:
            filmed_tedx[d.film_date.year] = filmed_tedx.get(d.film_date.year, 0) + 1
        filmed[d.film_date.year] = filmed.get(d.film_date.year, 0) + 1
        published[d.published_date.year] = published.get(d.published_date.year, 0) + 1

    pyplot.grid()
    pyplot.xticks(range(min(filmed.keys()), max(filmed.keys()) + 1), rotation=90)
    pyplot.plot([k for k in sorted(filmed.keys())], [filmed[k] for k in sorted(filmed.keys())],
                'o-', label='Talks filmed')
    pyplot.plot([k for k in sorted(published.keys())], [published[k] for k in sorted(published.keys())],
                'o-', label='Talks published')
    pyplot.plot([k for k in sorted(filmed_tedx.keys())], [filmed_tedx[k] for k in sorted(filmed_tedx.keys())],
                'o-', label='TEDx talks filmed')
    pyplot.legend()
    pyplot.show()


# frequency_over_years()


def time_to_publish():
    cut = 200
    x_interval = 250
    y_interval = 2

    d_film = sorted(ted_data, key=lambda d: d.film_date)[:]
    y_film = [dates.date2num(d.film_date) / 365 for d in d_film]
    y_publish = [dates.date2num(d.published_date) / 365 for d in d_film]
    tstamps = set([int(y) for y in y_film + y_publish])

    pyplot.grid(axis='y')
    pyplot.xticks(range(0, len(d_film) + x_interval, x_interval))
    pyplot.yticks(range(min(tstamps), max(tstamps) + y_interval, y_interval))
    pyplot.plot(range(len(d_film)), y_film, '-', linewidth=0.75)
    pyplot.plot(range(len(d_film)), y_publish, '-', linewidth=0.75)
    pyplot.fill_between(range(len(d_film)), y_film, y_publish, facecolor='xkcd:ice blue')
    # pyplot.axvline(cut, color='black')
    pyplot.ticklabel_format()
    pyplot.show()


# time_to_publish()

# print([d.published_date > d.film_date for d in data].count(False))


def language_histogram():
    interval = 1
    d = [d.languages for d in ted_data]
    bins = range(0, int(max(d)), interval)
    pyplot.grid(axis='y', zorder=0)
    pyplot.xticks(bins[::5])
    pyplot.hist(d, bins=bins, edgecolor='black', linewidth=0.5, zorder=2)
    pyplot.axvline(np.mean(d), color='black')
    pyplot.show()


# language_histogram()


def language_view_scatter():
    d_lang = [d.languages for d in ted_data]
    d_view = [d.views for d in ted_data]
    print(np.corrcoef(d_lang, d_view))

    pyplot.scatter(d_lang, d_view)
    pyplot.show()


# language_view_scatter()


def duration_view_scatter():
    d_dur = [d.duration for d in ted_data]
    d_view = [d.views for d in ted_data]
    print(np.corrcoef(d_dur, d_view))

    pyplot.scatter(d_dur, d_view)
    pyplot.show()


# duration_view_scatter()

def occupation_group():
    occ_frequency = dict()
    occ_views = dict()
    for d in ted_data:
        for occ in d.speaker_occupation:
            occ_frequency[occ] = occ_frequency.get(occ, 0) + 1
            occ_views[occ] = occ_views.get(occ, 0) + d.views

    # print(len(occ_frequency))
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
    years = range(2006, 2018)
    tag_frequency = dict()
    tag_views = dict()
    for d in ted_data:
        if d.film_date.year > 2005:
            for tag in d.tags:
                if tag == 'TEDx':
                    continue
                tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
                tag_views[tag] = tag_views.get(tag, 0) + d.views
    top_tags = sorted(tag_frequency, key=tag_frequency.get, reverse=True)[:8]

    pyplot.xticks(range(len(years)), years)
    pyplot.grid(axis='x')

    for tag in top_tags:
        counts = [len([d for d in ted_data if d.film_date.year == year and tag in d.tags]) / len(
            [d for d in ted_data if d.film_date.year == year]) * 100 for year in years]
        pyplot.plot(counts, label=tag)

    pyplot.legend()
    pyplot.show()


# tag_trend()


def comments_histogram():
    interval = 50
    d = [d.comments for d in ted_data]
    bins = range(0, int(max(d)) + interval, interval)
    print(bins)
    pyplot.grid(axis='y', zorder=0)
    pyplot.xticks(bins[::10])
    pyplot.hist(d, bins=bins, edgecolor='black', linewidth=0.5, zorder=2)
    pyplot.axvline(np.mean(d), color='black')
    pyplot.show()


# comments_histogram()


def discuss_histogram():
    interval = 1
    d = [10000 * d.comments / d.views for d in ted_data]
    # bins = range(0, int(max(d)) + interval, interval)
    # print(bins)
    pyplot.grid(axis='y', zorder=0)
    # pyplot.xticks(bins[::])
    pyplot.hist(d, bins=100, edgecolor='black', linewidth=0.5, zorder=2)
    pyplot.axvline(np.mean(d), color='black')
    pyplot.show()


# discuss_histogram()


def duration_longwinded_scatter():
    d_dur = [d.duration for d in ted_data]
    d_long = [d.ratings['Longwinded'] for d in ted_data]
    d_long_n = [d.norm_ratings['Longwinded'] for d in ted_data]
    print(np.corrcoef(d_dur, d_long_n))

    # pyplot.scatter(d_dur, d_long, marker='.')
    pyplot.scatter(d_dur, d_long_n, marker='.')
    pyplot.axvline(25, color='black', linewidth=0.5)
    pyplot.axvline(47, color='black', linewidth=0.5)
    pyplot.show()


# duration_longwinded_scatter()


def tech_occupation():
    d = [d for d in ted_data if 'technology' in d.tags]
    print(len(d))
    tech_occ = dict()
    for d in d:
        for occ in d.speaker_occupation:
            tech_occ[occ] = tech_occ.get(occ, 0) + 1

    top_occs = sorted(tech_occ, key=tech_occ.get, reverse=True)[:10]

    pprint([(k, tech_occ[k]) for k in top_occs])


# tech_occupation()


def writer_tags():
    d = [d for d in ted_data if 'writer' in d.speaker_occupation]
    print(len(d))
    writer_tag = dict()
    for d in d:
        for tag in d.tags:
            if tag == 'TEDx':
                continue
            writer_tag[tag] = writer_tag.get(tag, 0) + 1

    top_tags = sorted(writer_tag, key=writer_tag.get, reverse=True)[:10]
    pprint([(k, writer_tag[k]) for k in top_tags])


# writer_tags()


def tags_graph():
    all_tags = reduce(set.union, [d.tags for d in ted_data], set()) - {'TEDx'}

    edges = dict((k, dict()) for k in all_tags)

    for d in ted_data:
        for a in d.tags:
            for b in d.tags:
                if a == b or a not in all_tags or b not in all_tags:
                    continue
                u, v = sorted([a, b])
                edges[u][v] = edges[u].get(v, 0) + 1

    graph = nx.Graph()
    cut = 80

    for u in edges:
        for v in edges[u]:
            if edges[u][v] >= cut:
                graph.add_node(u)
                graph.add_node(v)
                graph.add_edge(u, v, weight=edges[u][v])

    edge_width = [d['weight'] / cut for (u, v, d) in graph.edges(data=True)]

    pos = nx.kamada_kawai_layout(graph, scale=100)
    nx.draw_networkx_nodes(graph, pos, node_size=10000, alpha=0)
    nx.draw_networkx_edges(graph, pos, width=edge_width, edge_cmap=cm.get_cmap('Set1'),
                           edge_color=range(len(graph.edges)))
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color='w',
                            bbox=dict(boxstyle='round', ec='k', fc='k', alpha=1))

    pyplot.axis('off')
    pyplot.show()


tags_graph()
