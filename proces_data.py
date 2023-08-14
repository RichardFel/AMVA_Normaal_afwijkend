#%%
# Import modules
import pandas as pd
import numpy as np
import logging
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


# Logging
today = datetime.date.today()
logging.basicConfig(filename=f'Logging/logging_{today}.log', level = logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')
# file_details = pd.DataFrame(columns = ['Subject','Sensor_location','measurement',	'Flip_end'	,
#                                        'Label_start',	'Label_stop',	'Expected_samples',
#                                            	'Sensor_tot_samples',	'Sensor_sample_start'
# ])
# file_details.to_excel('Meta_data/file_details.xlsx', index=0)

# Settings
visualise = False
data_path = 'Raw_data'
sensor_loc = 1
Resample_freq = 10
file_details = pd.read_excel('Meta_data/file_details.xlsx')

def resample_data(sensor_data, sample_freq, Resample_freq):
    removingNumber = int(np.floor(len(sensor_data) - (len(sensor_data)/sample_freq)*Resample_freq))
    interval = len(sensor_data)/removingNumber
    cumsumInterval = np.cumsum(np.ones(removingNumber)*interval)
    removalInterval = [ int(x) for x in cumsumInterval ][:-5] 
    return sensor_data.drop(removalInterval)

#%%
for file in os.listdir(data_path):
    # Skip different files
    if not file.endswith('.TXT'):
        continue

    subject, location = file.split('_')
    sensor, measurement, _ = location.split('.')

    # Check if ankle sensor
    if sensor != '3':
        continue

    # Check if data already processed
    if os.path.isfile(f'Labelled_data/{subject}_{measurement}.csv'):
        continue

    # Print
    print(f'{subject}, sensor lacation {sensor}')
    
    # Load data
    sensor_data = pd.read_csv(f'{data_path}/{file}', names = ['Time', 'Ax', 'Ay', 'Az', 
                                                'Gx', 'Gy','Gz',
                                                'Mx', 'My', 'Mz'])
    sensor_data.drop(columns=['Mx', 'My', 'Mz'], inplace=True)
    try:
        labeled_data = pd.read_csv(f'Labels/{subject}.{measurement}.csv', names = ['Start','Eind','Duur', 'Activiteit',
                                                                                'Label','Extra kolom','1','Filmnummer','2','3'],
                                                                                sep=';', skiprows= 2)
    except FileNotFoundError:
        logging.warning(f'{file}: No labelled data')
        continue

    # Calc sample freq
    measurement_time = (sensor_data.iloc[-1,0]  - sensor_data.iloc[0,0]) / 1000
    sample_freq =  len(sensor_data) / measurement_time  

    # Drop if sample freq below 50
    if sample_freq < 10:
        logging.warning(f'{file}: sample frequency at {sample_freq}')
        continue

    # Drop every 5th element from data (12.5Hz -> 10Hz)
    sensor_data = resample_data(sensor_data, sample_freq, Resample_freq)

    # Load labelled data
    try:
        flip = labeled_data.dropna(subset=['Extra kolom']).iloc[1]
        flip_time = int(flip['Extra kolom'])
        start_label = datetime.datetime.strptime(flip['Eind'],'%H:%M:%S')
        start_time = start_label.hour * 3600 + start_label.minute * 60 + start_label.second
        labeled_data = labeled_data.dropna(subset=['Label'])
        end = datetime.datetime.strptime(labeled_data['Eind'].iloc[-1],'%H:%M:%S')
        end_time = end.hour * 3600 + end.minute * 60 + end.second
        expected_samples =  (end_time - flip_time) * Resample_freq
    except (ValueError, IndexError, TypeError):
        logging.warning(f'{file}: Label error')

    sensor_samples = len(sensor_data)
    # if sensor_samples < expected_samples:
    #     logging.warning(f'{file}: missing data')
    #     continue

    # find the flip in the data
    fig, ax = plt.subplots()
    ax.plot(np.arange(0,1800), sensor_data.iloc[:1800,1:4])
    x = plt.ginput(1)

    # New data
    # Detect flip
    new_sensor_data = sensor_data.iloc[int(x[0][0]):int(x[0][0])+int(expected_samples),:]

    # correct for difference flip and labeltiming
    start_labels = start_time - flip_time 
    new_sensor_data = new_sensor_data.iloc[int(start_labels*Resample_freq):,:]


    # Add labels to array
    tmp = np.zeros(len(new_sensor_data))
    start = labeled_data.loc[labeled_data['Start'] == flip['Eind']].index[0]
    labels = labeled_data.loc[start:]
    for idx, label in enumerate(labels['Label']):
        tmp[idx*50:idx*50+50] = label

    # Add labels to data
    labeled_sensor_data = new_sensor_data.reset_index(drop=True).join(pd.DataFrame(tmp, columns = ['Label']))

    if visualise:
        fig, ax = plt.subplots(2)
        ax[0].plot(labeled_sensor_data.loc[((labeled_sensor_data['Label'] >= 5) & (labeled_sensor_data['Label'] <= 8)),'Ax':'Az'].reset_index(drop=True))
        ax[0].set_title('Active')
        ax[1].plot(labeled_sensor_data.loc[((labeled_sensor_data['Label'] >= 2) & (labeled_sensor_data['Label'] <= 3)),'Ax':'Az'].reset_index(drop=True))
        ax[1].set_title('Inactive')
        plt.tight_layout()
        plt.show()

    # Save data
    labeled_sensor_data.to_csv(f'Labelled_data/{subject}_{measurement}.csv')

    # Save metadata
    file_details.loc[subject] = [subject, sensor, measurement, flip_time, start_time,end_time, expected_samples, sensor_samples, int(x[0][0])] 
    file_details.to_excel('Meta_data/file_details.xlsx', index=0)

    # Close plot
    plt.close('all')

#%%