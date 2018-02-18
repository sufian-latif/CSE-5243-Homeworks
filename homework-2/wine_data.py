import csv
from pprint import pprint


class WineData:
    headers = []
    HIGH = 'High'
    LOW = 'Low'

    def __init__(self, row):
        self.data = [float(x) for x in row[:-1]]
        self.quality = WineData.HIGH if float(row[-1]) > 5 else WineData.LOW


def read_data(file_name):
    with open(file_name, 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        WineData.headers = next(reader)
        data = []

        for row in reader:
            data.append(WineData([s.strip() for s in row]))

    return data


def main():
    data = read_data('winequality-red.csv')
    print(len(data))
    pprint(data[0].__dict__)
    pprint(WineData.headers)


if __name__ == '__main__':
    main()
