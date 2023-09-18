#%%
# Modules
import pandas as pd
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

# Settings
data_path = 'Labelled_data'

zitten = [2, 3, 4, 9, 14, 15, 16, 17]
fietsen = [23]
lopen = [5, 6, 7, 8, 18, 19, 22, 26, 10, 12, 13, 33]

# functions


def def_categories(x):
    if x in zitten:
        return 0
    elif x in fietsen:
        return 1
    elif x in lopen:
        return 2


for file in os.listdir(data_path):
    if file.startswith('.'):
        continue
    sensor_data = pd.read_csv(f'{data_path}/{file}', index_col=0)

    # Set labels to labels ML
    sensor_data['Categories'] = sensor_data['Label'].apply(def_categories)

    # Drop unused sections of data
    sensor_data = sensor_data.dropna(subset=['Categories'])
    sensor_data = sensor_data.drop(columns=['Time', 'Label'])

    # Save data to processed_data
    sensor_data.to_csv(f'Processed_data_3groups/{file}')

#%%
