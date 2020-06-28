# TrafficModel

## How to run?

#### Step 1: Process Image
    1. At first we must read the image and make csv based on the road and traffic intensity.
    2. We cannot read 30 days data and store it in a single file.
    3. We need to make multiple csv file. 
    4. We can do it using process_data.py file.
    5. I made the csv for all the 30 days data and saved in 6 csv files
    6. You will find the csv files in the following links: https://drive.google.com/drive/u/0/folders/1SYYDbFC28o30gmAleKq9uMeBvl2yQL0x
    
#### Step 2: Run the models
    1. Firstly, set every settings in the config file
    2. The config file strcture is:
   
        config = {
            'train': {
                'set0': {
                    'start_date': '01_11_2019',
                    'end_date': '04_11_2019',
                    'start_time': '12_34_01',
                    'end_time': '12_36_31',
                },
            },
            'test': {
                'set0': {
                    'start_date': '01_11_2019',
                    'end_date': '04_11_2019',
                    'start_time': '12_34_01',
                    'end_time': '12_35_01',
                }
            },
            'intensity_list_to_count': [3, 4],
            'is_intensity_continuous': True,
            'number_of_ignored_cell': 3,
            'intermediate_file_save': True,
            'sampling_rate': 2,
            'seq_len': 12,
            'pre_len': 10,
            'repeat': False,
            'is_continuous': False,
        }
        
        Every key depicts:
        a. 'train': set the train data set. Under train you can set(range) different set of data from different portion of 
            the dataset
            For Example, 
                'set0': {
                    'start_date': '01_11_2019',
                    'end_date': '04_11_2019',
                    'start_time': '12_34_01',
                    'end_time': '12_36_31',
                },
            Here, we have a set of data for training set. This data set started from 1st november to 4th nomvember 
            and time range is from 12:34 to 12:36.
            
            Similarly we add another data set like following:
            'train': {
                'set0': {
                    'start_date': '01_11_2019',
                    'end_date': '04_11_2019',
                    'start_time': '12_34_01',
                    'end_time': '12_36_31',
                },
                'set1': {
                    'start_date': '07_11_2019',
                    'end_date': '11_11_2019',
                },
            },
            
            We already discussed about set0 and for the set1, program will filter the data from 7th november to 11th
            nomvember but what will be the time range in this set. Well, the time range will be the default one. 
            (06:00:01 to 23:59:31)
            
        b. 'test': Set multiple dataset(range) for the test data in the similar way.
        
        c. 'intensity_list_to_count': [3, 4],
        
        d. 'is_intensity_continuous': True,
        
        e. 'number_of_ignored_cell': 3,
        
        f. 'intermediate_file_save': True,
        
        g. 'sampling_rate': 2,
        
        h. 'seq_len': 12,
        
        i. 'pre_len': 10,
        
        j. 'repeat': False,
        
        k. 'is_continuous': False,
        
            
            
## Project Structure

![Project Structure](https://github.com/ResearchWithMahbubSir/TrafficModel/blob/master/Screen%20Shot%202020-04-18%20at%203.10.35.png)


Put images by creating a folder and write the folder name in main.py.
