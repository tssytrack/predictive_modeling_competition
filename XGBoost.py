#%% Import package
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import impute
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

#%% Reading Data
Training = pd.read_csv("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/MLProject_train.csv")
Validation = pd.read_csv("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/MLProject_valid.csv")
Data = Training.append(Validation, ignore_index = True)

Testing = pd.read_csv("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/MLProject_test.csv")

#%% Preprocess the data (missing imputing, standardizing, and one-hot encoding)

# to do standardization we firstly combine the training and validation sets together
# seperate numerical and categorical data
X = Data.drop(["Z2", "target1", "target2"], axis = 1).values
Y = Data.loc[:, ["target1", "target2"]].values
Z2 = np.array(Data.Z2).reshape(-1, 1)

# imputing numerical variables with median
imputer = impute.SimpleImputer(strategy = "median")
imputer.fit(X)
Imputed = imputer.transform(X)

# standardize the numerical variables
scaler = StandardScaler()
scaler.fit(Imputed)
Standardized = scaler.transform(Imputed)
Matrix = np.concatenate((Standardized, Z2, Y), axis = 1)

# Put the standardized and imputed data together
Variables = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'I1', 'J1', 'K1',
       'L1', 'M1', 'N1', 'P1', 'Q1', 'R1', 'S1', 'T1', 'U1', 'V1', 'W1',
       'X1', 'Y1', 'Z1', 'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2',
       'I2', 'J2', 'K2', 'L2', 'M2', 'N2', 'P2', 'Q2', 'R2', 'S2', 'T2',
       'U2', 'V2', 'W2', 'X2', 'Y2', 'A3', 'B3', 'C3', 'D3', 'E3',
       'F3', 'G3', 'H3', 'I3', 'J3', 'K3', 'L3', 'M3', 'N3', 'P3', 'Q3',
       'R3', 'S3', 'T3', 'U3', 'V3', 'W3', 'X3', 'Y3', 'Z3', 'A4', 'B4',
       'C4', 'D4', 'E4', 'F4', 'G4', 'H4', 'I4', 'J4', 'K4', 'L4', 'M4',
       'N4', 'P4', 'Q4', 'R4', 'S4', 'T4', 'U4', 'V4', 'W4', 'X4', 'Y4',
       'Z4', 'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5', 'I5', 'J5',
       'K5', 'L5', 'M5', 'N5', 'P5', 'Q5', 'R5', 'S5', 'T5', 'U5', 'V5',
       'W5', 'X5', 'Y5', 'Z5', 'A6', 'B6', 'C6', 'D6', 'E6', 'F6', 'G6',
       'H6', 'I6', 'J6', 'K6', 'L6', 'M6', 'N6', 'P6', 'Q6', 'R6', 'S6',
       'T6', 'U6', 'V6', 'W6', 'X6', "Z2", "Target1", "Target2"]

DataImputed = pd.DataFrame(Matrix, columns = Variables)

DataImputed["Z2"] = DataImputed["Z2"].round().astype("int")
DataImputed["Target1"] = DataImputed["Target1"].round().astype('int')
DataImputed["Target2"] = DataImputed["Target2"].round().astype('int')

# One hot encoding
Encoded = pd.get_dummies(DataImputed, prefix_sep="_", columns = ["Z2"])

# put the columns in the right order
cols = Encoded.columns.tolist()
cols = cols[:-7] + cols[-5:] + cols[-7:-5]
DataEncoded = Encoded.loc[:, cols]

# Change all the categorical variable including targets to boolean
DataEncoded.iloc[:, -7:] = DataEncoded.iloc[:, -7:].astype("bool")

# Creating Training and Validation sets
TRAIN = DataEncoded.iloc[:Training.shape[0] , :]
TEST = DataEncoded.iloc[Training.shape[0]:, :]

#%% Training and Testing Split
x_train, x_test, y1_train, y1_test = TRAIN.iloc[:, :-2], TEST.iloc[:, :-2], TRAIN.iloc[:, -2], TEST.iloc[:, -2]
y2_train, y2_test = TRAIN.iloc[:, -1], TEST.iloc[:, -1]

#%% Setting initial parameters for hyperparameter tunning
params_y1 = {"objective": "binary:logistic",
          "colsample_bytree": 1,
          "learning_rate" : 0.1,
          "max_depth": 5,
          "alpha": 10,
          "subsample": 1,
          "min_child_weight": 1}
params_y1["eval_metric"] = "auc"
num_boost_round = 500

# Creating xgboost dataset
dtrain = xgb.DMatrix(x_train, label = y1_train)
dtest = xgb.DMatrix(x_test, label = y1_test)

#%% max_depth and min_child_weight tunning
GridSearch = [
       (max_depth, min_child_weight)
       for max_depth in [4, 5]
       for min_child_weight in [2, 4]
]

max_depth_min_child_weight = []

max_auc = 0
best_max_depth = None
for max_depth, min_child_weight in GridSearch:
       print(f"CV with max_depth = {max_depth}, min_child_weight = {min_child_weight}")

       params_y1["max_depth"] = max_depth
       params_y1["min_child_weight"] = min_child_weight

       model = xgb.train(
              params_y1,
              dtrain,
              num_boost_round = num_boost_round,
              evals = [(dtest, "Test")],
              early_stopping_rounds = 10
       )

       # update auc
       auc = model.best_score
       boost_round = model.best_iteration
       print(f"\tAUC {auc} for {boost_round} rounds")
       max_depth_min_child_weight.append((auc, boost_round, max_depth, min_child_weight))
       if auc > max_auc:
              max_auc = auc
              best_depth_child_weight = (max_depth, min_child_weight)

print(f"Best params: {best_depth_child_weight[0]}, {best_depth_child_weight[1]}, AUC: {max_auc}")

params_y1["max_depth"] = best_depth_child_weight[0]
params_y1["min_child_weight"] = best_depth_child_weight[1]

#%% subsample and colsample_bytree
GridSearch = [
    (subsample, colsample)
    for subsample in [0.2, 0.5, 0.8]
    for colsample in [0.2, 0.5, 0.8]
]

subsample_colsample_y1 = []

max_auc = 0
best_subcolsample = None
for subsample, colsample in (GridSearch):
       print(f"CV with subsample = {subsample}, colsample_bytree = {colsample}")

       params_y1["subsample"] = subsample
       params_y1["colsample_bytree"] = colsample

       model = xgb.train(
              params_y1,
              dtrain,
              num_boost_round = num_boost_round,
              evals = [(dtest, "Test")],
              early_stopping_rounds = 10
       )

       # update auc
       auc = model.best_score
       boost_round = model.best_iteration
       print(f"\tAUC {auc} for {boost_round} rounds")
       subsample_colsample_y1.append((auc, boost_round, subsample, colsample))
       if auc > max_auc:
              max_auc = auc
              best_subcolsample = (subsample, colsample)

print(f"Best params: {best_subcolsample[0]}, {best_subcolsample[1]}, AUC: {max_auc}")

params_y1["subsample"] = best_subcolsample[0]
params_y1["colsample_bytree"] = best_subcolsample[1]

#%% tuning
max_auc = 0
best_auc_rate = None

LearningRate = []

for rate in [.3, .2, .1, .05, .01, .005]:
       print(f"learning rate = {rate}")

       params_y1["learning_rate"] = rate

       model = xgb.train(
              params_y1,
              dtrain,
              num_boost_round=num_boost_round,
              evals=[(dtest, "Test")],
              early_stopping_rounds=10
       )

       # update auc
       auc = model.best_score
       boost_round = model.best_iteration
       print(f"\tAUC {auc} for {boost_round} rounds")
       LearningRate.append((auc, boost_round, rate))
       if auc > max_auc:
              max_auc = auc
              best_auc_rate = rate

print(f"Best learning rate: {best_auc_rate}, AUC: {max_auc}")

params_y1["learning_rate"] = best_auc_rate
params_y1_final = {"objective": "binary:logistic",
                     "colsample_bytree": 0.8,
                     "learning_rate" : 0.1,
                     "max_depth": 5,
                     "alpha": 10,
                     "subsample": 0.5,
                     "min_child_weight": 2,
                     "eval_metric": "auc"}

#%% run with hyperparameters
final_model_y1 = xgb.train(
              params_y1,
              dtrain,
              num_boost_round=num_boost_round,
              evals=[(dtest, "Test")],
              early_stopping_rounds=100
       )

# Save the model
# final_model_y1.save_model("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/XGBoost_y1.model")
#
# final_model_y1 = xgb.Booster()
# final_model_y1.load_model("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/XGBoost_y1.model")

# Get PPV
y1_prediction = final_model_y1.predict(dtest)

y1_list = []
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
       y1_prediction = final_model_y1.predict(dtest)
       y1_prediction[y1_prediction >= i] = 1
       y1_prediction[y1_prediction < i] = 0
       ppv_y1 = precision_score(y1_test, y1_prediction)
       y1_list.append((i, ppv_y1))


ppv_y1 = precision_score(y1_test, y1_prediction)

max_ppv_y1 = max(y1_list, key = lambda item:item[1])

#%% parameter tunning for y2

params_y2 = {"objective": "binary:logistic",
          "colsample_bytree": 1,
          "learning_rate" : 0.1,
          "max_depth": 5,
          "alpha": 10,
          "subsample": 1,
          "min_child_weight": 1}
params_y2["eval_metric"] = "auc"
num_boost_round = 50

dtrain_y2 = xgb.DMatrix(x_train, label = y2_train)
dtest_y2 = xgb.DMatrix(x_test, label = y2_test)

#%% max_depth and min_child_wieght on Y2
gridsearch_params = [
       (max_depth, min_child_weight)
       for max_depth in [2, 3, 5]
       for min_child_weight in [1, 3, 5 ]
]
depth_child = []
max_auc = 0
best_params = None
for max_depth, min_child_weight in gridsearch_params:
       print(f"CV with max_depth = {max_depth}, min_child_weight = {min_child_weight}")

       # update our parameters
       params_y2["max_depth"] = max_depth
       params_y2["min_child_weight"] = min_child_weight

       model = xgb.train(
              params_y2,
              dtrain_y2,
              num_boost_round = num_boost_round,
              evals = [(dtest_y2, "Test")],
              early_stopping_rounds = 10
       )

       # update auc
       auc = model.best_score
       boost_round = model.best_iteration
       print(f"\tAUC {auc} for {boost_round} rounds")
       depth_child.append((max_depth, min_child_weight, auc, boost_round))
       if auc > max_auc:
              max_auc = auc
              best_params = (max_depth, min_child_weight)

print(f"Best params: {best_params[0]}, {best_params[1]}, AUC: {max_auc}")

params_y2["max_depth"] = best_params[0]
params_y2["min_child_weight"] = best_params[1]

#%% subsample and colsample_bytree on Y2

sub_col_sample = [
    (subsample, colsample)
    for subsample in [0.2, 0.5, 0.8]
    for colsample in [0.2, 0.5, 0.8]
]

sub_col = []

max_auc = 0
best_subcolsample = None

for subsample, colsample in reversed(sub_col_sample):
       print(f"CV with subsample = {subsample}, colsample = {colsample}")

       # update our parameters
       params_y2["subsample"] = subsample
       params_y2["colsample_bytree"] = colsample

       # Train
       model = xgb.train(
              params_y2,
              dtrain_y2,
              num_boost_round=num_boost_round,
              evals=[(dtest_y2, "Test")],
              early_stopping_rounds=10
       )

       # update auc
       auc = model.best_score
       boost_round = model.best_iteration
       print(f"\tAUC {auc} for {boost_round} rounds")
       sub_col.append((subsample, colsample, auc, boost_round))
       if auc > max_auc:
              max_auc = auc
              best_subcolsample = (subsample, colsample)

print(f"Best subsample, colsample_bytree: {best_subcolsample[0]}, {best_subcolsample[1]}, AUC: {max_auc}")

params_y2["subsample"] = best_subcolsample[0]
params_y2["colsample_bytree"] = best_subcolsample[1]

#%% tuning learning rate on Y2
max_auc = 0
best_rate = None

best_learning_rate_y2 = []

for rate in [.3, .2, .1, .05, .01, .005]:
       print(f"learning rate = {rate}")

       params_y2["learning_rate"] = rate

       model = xgb.train(
              params_y2,
              dtrain_y2,
              num_boost_round=num_boost_round,
              evals=[(dtest_y2, "Test")],
              early_stopping_rounds=10
       )

       # update auc
       auc = model.best_score
       boost_round = model.best_iteration
       print(f"\tAUC {auc} for {boost_round} rounds")
       best_learning_rate_y2.append((rate, auc))
       if auc > max_auc:
              max_auc = auc
              best_auc_rate = rate

print(f"Best learning rate: {best_auc_rate}, AUC: {max_auc}")

params_y2["learning_rate"] = best_auc_rate
params_y2_final = {"objective": "binary:logistic",
                     "colsample_bytree": 0.5,
                     "learning_rate" : 0.1,
                     "max_depth": 3,
                     "alpha": 10,
                     "subsample": 0.8,
                     "min_child_weight": 1,
                     "eval_metric": "auc"}

#%% run with hyperparameters y2
final_model_y2 = xgb.train(
              params_y2,
              dtrain_y2,
              num_boost_round=num_boost_round,
              evals=[(dtest_y2, "Test")],
              early_stopping_rounds=100
       )

# Save the model
# final_model_y2.save_model("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/XGBoost_y2.model")

# final_model_y1 = xgb.Booster()
# final_model_y1.load_model("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/XGBoost_y1.model")

# Get PPV
y2_prediction = final_model_y2.predict(dtest_y2)

y2_list = []
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
       y2_prediction = final_model_y2.predict(dtest_y2)
       y2_prediction[y2_prediction >= i] = 1
       y2_prediction[y2_prediction < i] = 0
       ppv_y2 = precision_score(y2_test, y2_prediction)
       y2_list.append((i, ppv_y2))


ppv_y2 = precision_score(y2_test, y2_prediction)

max_ppv_y2 = max(y2_list, key = lambda item:item[1])

#%% Process Testing set for making final prediction

# Checking missing values for all the columns in the Testing set
TestingMissing = Testing.isnull().sum()
TestingMissing[TestingMissing == 0].shape

# No missing values in the testing dataset, but we still need to standardize the data and do the one-hot encoding
testing_numeric = Testing.drop("Z2", axis = 1).values
testing_stand = pd.DataFrame(scaler.transform(testing_numeric))
testing_stand["Z2"] = Testing["Z2"].round()
testing_stand = pd.get_dummies(testing_stand, prefix_sep = "_", columns = ["Z2"])
testing_stand.iloc[:, -5:] = testing_stand.iloc[:, -5:].astype("bool")

testing_columns = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'I1', 'J1', 'K1',
       'L1', 'M1', 'N1', 'P1', 'Q1', 'R1', 'S1', 'T1', 'U1', 'V1', 'W1',
       'X1', 'Y1', 'Z1', 'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2',
       'I2', 'J2', 'K2', 'L2', 'M2', 'N2', 'P2', 'Q2', 'R2', 'S2', 'T2',
       'U2', 'V2', 'W2', 'X2', 'Y2', 'A3', 'B3', 'C3', 'D3', 'E3',
       'F3', 'G3', 'H3', 'I3', 'J3', 'K3', 'L3', 'M3', 'N3', 'P3', 'Q3',
       'R3', 'S3', 'T3', 'U3', 'V3', 'W3', 'X3', 'Y3', 'Z3', 'A4', 'B4',
       'C4', 'D4', 'E4', 'F4', 'G4', 'H4', 'I4', 'J4', 'K4', 'L4', 'M4',
       'N4', 'P4', 'Q4', 'R4', 'S4', 'T4', 'U4', 'V4', 'W4', 'X4', 'Y4',
       'Z4', 'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5', 'I5', 'J5',
       'K5', 'L5', 'M5', 'N5', 'P5', 'Q5', 'R5', 'S5', 'T5', 'U5', 'V5',
       'W5', 'X5', 'Y5', 'Z5', 'A6', 'B6', 'C6', 'D6', 'E6', 'F6', 'G6',
       'H6', 'I6', 'J6', 'K6', 'L6', 'M6', 'N6', 'P6', 'Q6', 'R6', 'S6',
       'T6', 'U6', 'V6', 'W6', 'X6', "Z2_1", "Z2_2", "Z2_3", "Z2_4", "Z2_5"]

testing_stand.columns = testing_columns

# Once the testing dataset is ready, we can have the final prediction on the testing set
Dtesting = xgb.DMatrix(testing_stand)

final_model_y1 = xgb.Booster()
final_model_y1.load_model("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/XGBoost_y1.model")
y1_final = final_model_y1.predict(Dtesting)

final_model_y2 = xgb.Booster()
final_model_y2.load_model("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/XGBoost_y2.model")
y2_final = final_model_y2.predict(Dtesting)

row = np.array(range(len(y1_final)))

# Submission csv
Submission = pd.DataFrame({"row": row, "target1": y1_final, "target2": y2_final})
Submission.to_csv("/Users/dauku/Desktop/Courses/2019Fall/Machine Learning/FinalProject/Final_Submission.csv", index = False)

# xgb.plot_importance(final_model_y1)
# plt.rcParams["figure.figsize"] = [5, 5]
# plt.show()