# %%
import pandas as pd
import numpy as np

# This function is for reading the file data and data labels.
def get_human_data(filename):
    data = pd.read_csv(filename, sep=';')
    y_label = data['Story']
    emotions = ['Emo.Anxiety','Emo.Despair','Emo.Irritation','Emo.Rage',]
    full_data = data[emotions].values
    return full_data,y_label

def get_human_limit_data(filename):
    data = pd.read_csv(filename, sep=';')
    y_label = data['Story']
    emotions = ['mc.Anxiety','mc.Despair','mc.Irritation','mc.Rage',]
    full_data = data[emotions].values
    return full_data,y_label

# This function is for extracting the precision list
# The list extract data when its feature and the label are the same
def get_precision_list(data,label):
    precision_list = []
    target_names = ['Anxiety','Despair','Irritation','Rage']
    for i in range(len(data)):
        index = target_names.index(label[i])
        precision_list.append(data[i][index])
    return precision_list

human_data_file = 'data/Exp2_human.csv'
human_data, human_data_label= get_human_data(human_data_file)
human_data_limit, human_data_label_limit = get_human_limit_data(human_data_file)
# %%
precision_list = get_precision_list(human_data, human_data_label)
precision_list_limit = get_precision_list(human_data_limit, human_data_label_limit)

mean = np.average(precision_list)
variance = np.var(precision_list)

mean_limit = np.average(precision_list_limit)
variance_limit = np.var(precision_list_limit)

print("Exp2 human free-rating data mean and variance are ", mean,variance)
# 0.29215564961029417 0.0067888217049672
print("Exp2 human limit-rating data mean and variance are ", mean_limit,variance_limit)
# 0.6617647058823529 0.22383217993079588
#%%
