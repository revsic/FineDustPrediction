import csv
import numpy as np

# 9 x 8 matrix, 25 city
SEOUL_MAP = [
    [0, 0, 0, 0, '도봉구', '노원구', 0, 0],
    [0, 0, 0, '강북구', '강북구', '노원구', '노원구', 0],
    [0, '은평구', '종로구', '성북구', '성북구', '성북구', '중랑구', 0],
    [0, '은평구', '서대문구', '종로구', '종로구', '동대문구', '중랑구', 0],
    [0, '은평구', '서대문구', '서대문구', '중구', '성동구', '광진구', '강동구'],
    [0, '마포구', '마포구', '마포구', '용산구', '강남구', '송파구', '강동구'],
    ['강서구', '강서구', '영등포구', '동작구', '서초구', '강남구', '송파구', 0],
    [0, '양천구', '영등포구', '관악구', '서초구', '강남구', '송파구', 0],
    [0, '구로구', '금천구', '관악구', '서초구', 0, 0, 0]
]


def read_csv(name, num_region=39):
    """Read the csv file and return the pm10 data
    
    Args:
        name: str, the name of the csv file.
        num_region: int, the number of the region including city and road.

    Returns:
        dist, dust data grouped by date and city.
        {'20170506': {'강남구': 50, '강동구': 60, '강서구':70}, '20170507': {'강남구': } ...}
    
    Raises:
        ValueError: If the number of region in the csv file is not equal to `num_region`.
    """
    with open(name, encoding='utf-8') as f:
        raw = csv.reader(f)
        raw.__next__()

        date = ''
        group = []
        group_by_date = []

        for i, line in enumerate(raw):
            if i % num_region == 0:
                group_by_date.append([])
                group = group_by_date[-1]

                date = line[0]

            if date == line[0]:
                group.append(line)
            else:
                raise ValueError(date + 'is not same as ' + line[0])

    group_by_city = {}
    for regions in group_by_date:
        date = regions[0][0]
        group_by_city[date] = {}

        group = group_by_city[date]
        for region in regions:
            date, name, no2, o3, co, so2, pm10, pm20 = region
            if name.endswith('구'):
                if pm10 != '':
                    group[name] = int(pm10)
                else:
                    del group_by_city[date]
                    break

    return group_by_city


def geographical_mapping(dust_data, seoul_map=SEOUL_MAP):
    """Map dust data to geographical map of Seoul.
    
    Args:
        dust_data: dict, dust data grouped by date and city (from `read_csv`).
            {'20170506': {'강남구': 50, '강동구': 60, '강서구': 70}, '20170507': {'강남구': } ...}
        seoul_map: list of int, geographical map of Seoul.
    
    Returns:
        dict, dust data mapped by the geographical map of Seoul, `seoul_map`.
        {'20170506': [[0, 0, 0, 0, 50, 60, 0, 0], [0, 0, 0, 40, ...]]}
      
    """
    mapped = {}
    for date, data in dust_data.items():
        mapped[date] = []
        map_of_seoul = mapped[date]

        for line in seoul_map:
            transformed = list(map(lambda x: data[x] if x != 0 else 0, line))
            map_of_seoul.append(transformed)

    return mapped


def generate_dataset(data, len_time=7):
    """ Generate the dust dataset grouped by N days.
    
    Args:
        data: dict, dust data grouped by date and city (from `read_csv`)
            {'20170506': {'강남구': 50, '강동구': 60, '강서구': 70}, '20170507': {'강남구': } ...}
        len_time: int, length of the time step. dataset is grouped by `len_time` days.

    Returns:
        sampled: the amount of fine dust bundled in 7 days.
        result: the amount of fine dust on the next 7 days.
    """
    mapped = geographical_mapping(data)
    
    # tie data to WEEKLY_BATCH(7) batches
    data = list(map(lambda x: x[1], sorted(mapped.items())))
    tied = list(map(lambda i: data[i:i+len_time], range(len(data) - len_time)))

    sampled = tied[:-1]
    result = tied[1:]
    
    return sampled, result


def create_dataset(csv_name='./MonthlyAverageAirPollutionInSeoul.csv',
                   len_time=7,
                   train_rate=0.7):
    data = read_csv(csv_name)
    sampled, result = generate_dataset(data, len_time)

    data_set = list(zip(sampled, result))
    np.random.shuffle(data_set)

    train_size = int(len(data_set) * train_rate)
    train_set = data_set[:train_size]
    train_sampled = list(map(lambda x: x[0], train_set))
    train_result = list(map(lambda x: x[1], train_set))

    test_set = data_set[-train_size:]
    test_sampled = list(map(lambda x: x[0], test_set))
    test_result = list(map(lambda x: x[1], test_set))

    return (train_sampled, train_result), (test_sampled, test_result)
