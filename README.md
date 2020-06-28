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
        a. 'train': set the train data set. Under train you can set(range) different set of data 
            from different portion of the dataset
            For Example, 
                'set0': {
                    'start_date': '01_11_2019',
                    'end_date': '04_11_2019',
                    'start_time': '12_34_01',
                    'end_time': '12_36_31',
                },
            Here, we have a set of data for training set. This data set started from 1st november
            to 4th nomvember and time range is from 12:34 to 12:36.
            
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
            
            We already discussed about set0 and for the set1, program will filter the data from 7th
            november to 11th nomvember but what will be the time range in this set. Well, the time 
            range will be the default one. (06:00:01 to 23:59:31)
            
        b. 'test': Set multiple dataset(range) for the test data in the similar way of train data.
        
        c. 'intensity_list_to_count': [3, 4],
            Lets say, we have some road  to intersection 'y, then when we calcluate the value
            of  intersection 'y' using these roads which comes to that intersection, we will only 
            consider the traffice intensity of those road pixels whose intensity exists in this 
            list('intensity_list_to_count': [3, 4]).
            
        d. 'is_intensity_continuous': True,
            Lets say, 
                      x                 y
                    [ 3, 4, 0, 0, 1, 4, 3] -> Y
             
             Here x to y is a road, which is connected to intersection Y. How much intensity will be added 
             th intersection Y for this calue ?
             
            If 'is_intensity_continuous': False, and  'intensity_list_to_count': [3, 4],
                we will calculated  the number of intensity with 3 or 4 over the road and add with the intersection.
            but if 'is_intensity_continuous': True,
                then we will not consider the previous pixels once we get pixels which intensity is not belong to 
                the set 'intensity_list_to_count': [3, 4]. For example, from set [ 3, 4, 0, 0, 1, 4, 3],  we will add
                2 with the intersection. We will not consider first (3, 4) as  after first (3, 4) we get some 
                intensities(0, 0 ,1) which does not belong to 'intensity_list_to_count': [3, 4].
        
        e. 'number_of_ignored_cell': 3, (Work only if 'is_intensity_continuous' is True)
            In the ('is_intensity_continuous': True) part, I describe about how we will count the intensity for an
            intersection if 'is_intensity_continuous' is set to true. The "number_of_ignored_cell" will define, at most 
            how many continous intensity can be ignored if they dont belong to list 'intensity_list_to_count': [3, 4],
            For example,
                [ 3, 4, 0, 0, 1, 4, 3]
                for this road wont add starting 3,4 with intersection if 'number_of_ignored_cell': 2. But if 
                'number_of_ignored_cell' is  3 then we can add starting (3, 4) with the intersection as we can ignore 
                 three (0, 0, 1).
                 
        f. 'intermediate_file_save': True,
            Wheather we will save the file  where each row will define each intersection and
            each column will define a timestamp.
    
        g. 'sampling_rate': 2,
            Currently we have data for every 30 seconds. If sampling rate is 2, we will create the sequences considering
            the data for every 60 seconds.
            
        h. 'seq_len': 12,
            The number of timestamp every sequence will hold.
        
        i. 'pre_len': 10,
            The number of prediction length. (How many timestamp the model will predict).
            
        j. 'repeat': False,
            If pre_len is 5 and repeat is true, then we will repeat the last 4 prediction with the first one.
            If repeat is false, we will calculate the next prediction after removing the first timestamp and 
            adding the current predicted data.  
        
        k. 'is_continuous': False,
            Lets say, we predicted  the next 5 timestamp. Now if is_continuous is True, then we will consider
            all the predicted timestamp to calculate the accuracy. Otherwise we will only use the last one.
          
            
## Project Structure

![Project Structure](https://github.com/ResearchWithMahbubSir/TrafficModel/blob/master/Screen%20Shot%202020-04-18%20at%203.10.35.png)


Put images by creating a folder and write the folder name in main.py.
