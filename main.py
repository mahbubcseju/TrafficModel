from config import config

for key in config['train']:
    value = config['train'][key]
    start_date = value['start_date'] if 'start_date' in value else '01_11_2019'
    end_date = value['end_date'] if 'end_date' in value else '01_11_2019'
    start_time = value['start_time'] if 'start_time' in value else '01_11_2019'
    end_time = value['end_time'] if 'end_time' in value else '01_11_2019'

    print(start_date, end_date, start_time, end_time)


for key in config['test']:
    value = config['test'][key]
    start_date = value['start_date'] if 'start_date' in value else '01_11_2019'
    end_date = value['end_date'] if 'end_date' in value else '01_11_2019'
    start_time = value['start_time'] if 'start_time' in value else '01_11_2019'
    end_time = value['end_time'] if 'end_time' in value else '01_11_2019'

    print(start_date, end_date, start_time, end_time)