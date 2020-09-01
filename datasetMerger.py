'''
@file       datasetMerger.py
@date       2020/09/01
@brief      Script to merge datasets for Team 1 and Team 2
'''

import pandas as pd

# Function to turn a space-separated string into a list
def string2List(text):
    return text.split(' ')

# Read data from both files
dataset1 = pd.read_csv('datasets/Team1Dataset.csv')
dataset2 = pd.read_csv('datasets/Team2Dataset.csv')

# Rename columns in dataset 2 and turn strings of tags into lists
dataset2.columns = ['Topic Title', 'Leading Comment', 'Tags']
dataset2['Tags'] = dataset2['Tags'].apply(lambda x: string2List(x))

# Join and save the two dataframes
merged = pd.concat([dataset1, dataset2], join='inner')
merged.to_csv('datasets/MergedDataset.csv')

