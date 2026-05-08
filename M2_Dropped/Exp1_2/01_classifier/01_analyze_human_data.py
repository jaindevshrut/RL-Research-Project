# This file includes two ways of analysising human data.
# precision list & entropy list

# %%
import pandas as pd
import numpy as np
# %%


# This function is for reading the file data and data labels.
def get_human_data(filename):
    data = pd.read_csv(filename, sep=';')
    y_label = data['Story']
    emotions = ['Emo.Happiness','Emo.Joy','Emo.Pride','Emo.Boredom',
        'Emo.Fear','Emo.Sadness','Emo.Shame',]
    full_data = data[emotions].values
    return full_data, y_label

def get_human_limit_data(filename):
    data = pd.read_csv(filename, sep=';')
    y_label = data['Story']
    emotions = ['Em.Happiness','Em.Joy','Em.Pride','Em.Boredom',
        'Em.Fear','Em.Sadness','Em.Shame',]
    full_data = data[emotions].values
    return full_data, y_label

# This function is for extracting the precision list
# The list extract data when its feature and the label are the same

def get_precision_list(data,label):
    precision_list = []
    target_names = ['Happiness','Joy','Pride','Boredom','Fear','Sadness','Shame']
    for i in range(len(data)):
        index = target_names.index(label[i])
        precision_list.append(data[i][index])
    return precision_list

human_data_file = 'data/Exp1_human.csv'
human_data, human_data_label= get_human_data(human_data_file)

precision_list = get_precision_list(human_data, human_data_label)
mean = np.average(precision_list)
variance = np.var(precision_list)

print(mean,variance)
# For human_free:
# 0.4019302024931973 0.027696655833540938

#%%
# To calculate the precision for human limited rating data
human_limit_data_file = 'data/human_limit.csv'
human_limit_data, human_limit_data_label= get_human_limit_data(human_limit_data_file)

precision_list_limit = get_precision_list(human_limit_data, human_limit_data_label)
mean_limit = np.average(precision_list_limit)
variance_limit = np.var(precision_list_limit)

print(mean_limit,variance_limit)
# For human_limit:
# 0.8714285714285714 0.11204081632653061