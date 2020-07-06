config = {
    'train': {
        'set0': {
            'start_date': '01_11_2019',
            'end_date': '15_11_2019',
            # 'start_time': '06_00_01',
            # 'end_time': '23_59_31',
        },
    },
    'test': {
        'set0': {
            'start_date': '16_11_2019',
            'end_date': '20_11_2019',
            # 'start_time': '0_34_01',
            # 'end_time': '22_35_01',
        }
    },
    'intensity_list_to_count': [3, 4],
    'is_intensity_continuous': True,
    'number_of_ignored_cell': 3,
    'intermediate_file_save': False,
    'sampling_rate': 2,
    'seq_len': 12,
    'pre_len': 10,
    'repeat': False,
    'is_continuous': False,
}
