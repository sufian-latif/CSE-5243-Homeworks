import csv
import pprint
import re
import datetime


class TEDData:
    headers = []

    def __init__(self, row):
        self.comments = int(row[0])
        self.desc = row[1]
        self.duration = int(row[2]) / 60.0
        self.event = row[3]
        self.film_date = datetime.datetime.fromtimestamp(int(row[4]))
        self.languages = int(row[5])
        self.main_speaker = row[6]
        self.name = row[7]
        self.num_speaker = row[8]
        self.published_date = datetime.datetime.fromtimestamp(int(row[9]))
        self.ratings = dict((r['name'], r['count']) for r in eval(row[10]))
        self.norm_ratings = dict((k, self.ratings[k] / sum(self.ratings.values())) for k in self.ratings)
        self.related_talks = eval(row[11])
        self.speaker_occupation = [s.strip() for s in re.split('[,/;]', row[12].lower())]
        self.tags = eval(row[13])
        self.title = row[14]
        self.url = row[15]
        self.views = int(row[16])


def read_data(file_name):
    with open(file_name, 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile)
        TEDData.headers = next(reader)
        data = []

        for row in reader:
            data.append(TEDData([s.strip() for s in row]))

    return data


def main():
    data = read_data('ted_main.csv')
    print(len(data))
    pprint.pprint(data[0].__dict__)


if __name__ == '__main__':
    main()
