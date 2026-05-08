#%%
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# we have 70 rows of testing data. 7 emotion, 10 model result for each emotion. 
def read_data (data_file):
    data = pd.read_csv(data_file)
    X = data[['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power']].values
    y = data['Emotion'].values
    return X,y

def predict_with_svm(c,x_train, y_train, X_test):
    svc = svm.SVC(kernel='linear', C=c, probability=True).fit(x_train,y_train)
    predictions = [svc.predict_proba(x.reshape(1, -1)).tolist()[0] for x in X_test]
    return predictions

def plot_c_precision(min_value, max_value, X_train, y_train, X_test, y_test):
    c_values = np.linspace(min_value, max_value, 80)
    avg_correct_prediction_rates = []
    target_names = list(dict.fromkeys(y_train))

    for c in c_values:
        predictions = predict_with_svm(c, X_train, y_train, X_test)
        correct_prediction_rates = []

        for i, prediction in enumerate(predictions):
            target_index = target_names.index(y_test[i])
            correct_prediction_rates.append(prediction[target_index])

        avg_correct_prediction_rates.append(np.mean(correct_prediction_rates))

    plt.plot(c_values, avg_correct_prediction_rates)
    plt.xlabel('C value')
    plt.ylabel('Average correct prediction rate')
    plt.show()

x_training, y_training = read_data('data/classifier_train.csv')
x_testing, y_testing = read_data('data/classifier_test.csv')
plot_c_precision(0.002,0.005,x_training,y_training,x_testing)
#%%
# This is for plotting the classifier with a range of c, so that we can compare the y-axie
# and figure out the suitable c value range for fitting human data.
# From the human data (humandata_for_c.py), we get the result that 
# for precision, mean = 0.4019302024931973, var = 0.027696655833540938

# Here is to find c according to precision plot
# x is a range of c value, y is the precision

# for every c, we have 70 precision added from 70 rows, and then get the average precision



# For exp3 free-rating
# mean = 0.29215564961029417 var = 0.0067888217049672
# c=0.0013, var = 0.0001
# For exp3 limit rating
# mean = 0.6617647058823529 var = 0.22383217993079588
# c=0.0034, var = 0.001

#%%

# https://scikit-learn.org/stable/modules/svm.html
# https://people.revoledu.com/kardi/tutorial/Python/SVM+in+Python.html
