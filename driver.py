import os
import urllib
import urllib.request
import pandas as pd
import numpy as np
import plotnine as p9

import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from statistics import mode
import inspect
import warnings

from classifiers.knn import MyKNN
import classifiers.logistic_regression
import classifiers.naive_bayes
from classifiers.cv import MyCV

print( "Finished loading." )

# Display the title
print("\nCS 480: Final Project Start")
print("================================\n")

MyKNN_N_NEIGHBORS_VAL = 20
CV_VAL = 5

# MISC. VARIABLES
kf = KFold(n_splits=3, shuffle=True, random_state=1)
test_acc_df_list = []
pipe = make_pipeline( StandardScaler() )

# Suppress annoying plotnine warnings
warnings.filterwarnings('ignore')

data_file = 'C:/Users/richard/Documents/GitHub/CS470_FinalProject/data/spambase.csv'
spam_df = pd.read_csv( data_file, header=None, skiprows=1 )
spam_label_col = 57

spam_labels = spam_df[spam_label_col]
spam_data = spam_df.iloc[:, :56].to_numpy()
pipe.fit(spam_data, spam_labels)


# Create data dictionary
data_dict = {
    'spam' : [spam_data, spam_labels],
}

# Loop through each data set
for data_set, (input_data, output_array) in data_dict.items():
    # Output message for logging
    print("Working on set: " + str(data_set))
    current_set = str(data_set)
    # Scale the data set

    # Loop over each fold for each data set
    for foldnum, indicies in enumerate(kf.split(input_data)):
        print("Fold #" + str(foldnum))
        # Set up input data structs
        index_dict = dict(zip(["train", "test"], indicies))
        param_dicts = [{'n_neighbors':[x]} for x in range(1, 21)]

        logreg_param_grid = [{'max_iterations':max_it} \
                                for max_it in [100] \
                                for steps in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]]

        knn_param_grid = [{'n_neighbors':max_n} \
                                for max_n in [20]]

        # Establish different models
        clf = MyCV( estimator = classifiers.knn.MyKNN,
                    param_grid = knn_param_grid,
                    cv=CV_VAL )

        RegressionCV = MyCV(estimator=classifiers.logistic_regression.MyLogReg,
                            param_grid=logreg_param_grid,
                            cv=CV_VAL)

        # Creating dictionary with input and outputs
        set_data_dict = {}
        for set_name, index_vec in index_dict.items():
            set_data_dict[set_name] = {
                "X":input_data[index_vec],
                "y":output_array.iloc[index_vec].reset_index(drop=True)
            }

        # Train the models with given data
        clf.fit(**set_data_dict["train"])
        RegressionCV.fit(**set_data_dict["train"])

        # Get most common output from outputs for featureless set
        most_common_element = mode(set_data_dict["train"]['y'])

        # Get results
        #cv_df = pd.DataFrame( clf.cv_results_ )
        #cv_df.loc[:, ["param_n_neighbors", "mean_test_score"]]
        pred_dict = {
            "K Nearest Neighbors": \
                clf.predict(set_data_dict["test"]["X"]),
            "Logistic Regression": \
                RegressionCV.predict(set_data_dict["test"]["X"]),
            "Featureless":most_common_element
        }

        # Build results dataframe for each algo/fold
        for algorithm, pred_vec in pred_dict.items():
            test_acc_dict = {
                "test_accuracy_percent":(
                    pred_vec == set_data_dict["test"]["y"]).mean()*100,
                "data_set":data_set,
                "fold_id":foldnum,
                "algorithm":algorithm
            }
            test_acc_df_list.append(pd.DataFrame(test_acc_dict, index=[0]))

# Build accuracy results dataframe
test_acc_df = pd.concat(test_acc_df_list)

# Print results
print("\n")
print(test_acc_df)

# Plot results
plot = (p9.ggplot(test_acc_df,
                    p9.aes(x='test_accuracy_percent',
                    y='algorithm'))
               + p9.facet_grid('. ~ data_set')
               + p9.geom_point())

print(plot)

print("\nCS 480: Final Project Program End")
print("==============================\n")
