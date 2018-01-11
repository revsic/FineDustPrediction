import numpy as np

SEOUL_MAP = [ # 9 by 8 matrix, 25 borough
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

def read_csv(name, location_cnt=39):
    """Read the csv file and return the amount of fine dust for each region grouped by date.
    
    Args:
        name: The name of the csv file to read.
    
    Returns:
        Dictionary object mapping the amount of fine dust for each region by date.
        {'20170506': {'강남구': 50, '강동구': 60, '강서구':70}, '20170507': {'강남구': } ...}
    
    Raises:
        ValueError: If the number of the data per day in the csv file is not equal to LOCATION_CNT.
        
    """
    with open(name) as f:
        raw_data = f.read().strip()

    del_quote = raw_data.replace("\"", '')
    data = list(map(lambda x: x.split(','), del_quote.split('\n')))[1:] # [1:] csv header

    splitted = []

    ptr = 0
    for i in range(len(data) // location_cnt):
        splitted.append(data[ptr:ptr+location_cnt])
        ptr += location_cnt
    
    ## test case
    for date_list in splitted:
        date = date_list[0][0] # index 0:date
        for local in date_list:
            if date != local[0]:
                raise ValueError(date + ' is not same as ' + 'local[0]')
    
    def filter_borough(dic):
        return dict(filter(lambda t: '구' in t[0], dic.items())) #filter not road name only borough 

    # index 0:date, 1:local name, 6:pms
    pms = dict(map(lambda x: (x[0][0], dict(map(lambda t: (t[1], t[6]), x))), splitted))
    pms_filtered = dict(filter(lambda x: '' not in x[1].values(), pms.items())) # csv data contains spaces
    pms_filtered2 = dict(map(lambda x: (x[0], filter_borough(x[1])), pms_filtered.items()))
    
    return pms_filtered2

def geographical_mapping(pms_data):
    """Map the amount of fine dust for each region to geographical map of Seoul.
    
    Args:
        pms_data: Fine dust data pre-processed by read_csv.
            Dictionary obejct mapping the amount of fine dust for each region by date.
            {'20170506': {'강남구': 50, '강동구': 60, '강서구':70}, '20170507': {'강남구': } ...}
    
    Returns:
        Dictionary that map the amount of fine dust to the geographical map of Seoul.
        {'20170506': [[0, 0, 0, 0, 50, 60, 0, 0], [0, 0, 0, 40, ...]]}
      
    """
    def dict2seoul(p):
        return list(map(lambda t: list(map(lambda x: int(p[x]) if x != 0 else 0, t)), seoul_map))

    # map dict to seoul geographic map
    pms_mapped = dict(map(lambda p: (p[0], dict2seoul(p[1])), pms_data.items())) 
    return pms_mapped

def generate_dataset(data, weekly_batch=7):
    """ Generate the daily average amount of the fine dust(pm10) bundled in 7 days.
    
    Args:
        data: Fine dust data pre-processed by read_csv
            Dictionary object mapping the amount of fine dust for each region by date.
            {'20170506': {'강남구': 50, '강동구': 60, '강서구':70}, '20170507': {'강남구': } ...}
    
    Returns:
        pms_sampled: the amount of fine dust bundled in 7 days.
        pms_result: the amount of fine dust on the next 7 days.
        
    """
    pms_mapped = geographical_mapping(data)
    
    # tie data to WEEKLY_BATCH(7) batches
    pms_data = list(map(lambda x: x[1], sorted(pms_mapped.items()))) 
    pms_tied = list(map(lambda i: pms_data[i:i+weekly_batch], range(len(pms_data) - weekly_batch)))
    
    pms_sampled = pms_tied[:-1]
    pms_result = pms_tied[1:]
    
    return pms_sampled, pms_result

def main():
    csv_name = 'MonthlyAverageAirPollutionInSeoul.csv'
    pms_data = read_csv(csv_name)
    pms_sampled, pms_result = generate_dataset(pms_data)

    data_set = list(zip(pms_sampled, pms_result))
    np.random.shuffle(data_set)

    train_set = data_set[:TRAIN_SIZE]
    train_sampled = list(map(lambda x: x[0], train_set))
    train_result = list(map(lambda x: x[1], train_set))

    test_set = data_set[-TEST_SIZE:]
    test_sampled = list(map(lambda x: x[0], test_set))
    test_result = list(map(lambda x: x[1], test_set))

