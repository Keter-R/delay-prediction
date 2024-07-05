import numpy as np
import pandas as pd
import torch
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from DataLoader import DataModule
from imblearn.over_sampling import SMOTEN

from ModelModule import calculate_metrics


def load_from_xlsx():
    file_path = 'data/ttc-streetcar-delay-data-2019.xlsx'
    # read all sheets from the Excel file
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    dat = None
    target_cols = ['Report Date', 'Time', 'Day', 'Incident', 'Min Delay', 'Route', 'Line', 'Date', 'Station ID',
                   'Delay']
    # target_cols = ['Report Date', 'Time', 'Day', 'Incident', 'Min Delay', 'Date', 'Station ID', 'Delay']
    for sheet_name in sheet_names:
        df = pd.read_excel(file_path, sheet_name)
        # remove cols that not in target_cols
        for col in df.columns:
            if col not in target_cols:
                df = df.drop(columns=[col])
        # check if the column 'Delay' is in the DataFrame and change it to 'Min Delay'
        if 'Delay' in df.columns:
            df = df.rename(columns={'Delay': 'Min Delay'})
        # check if the column 'Route' is in the DataFrame and change it to 'Line'
        if 'Line' in df.columns:
            df = df.rename(columns={'Line': 'Route'})
        if 'Report Date' in df.columns:
            df = df.rename(columns={'Report Date': 'Date'})
        # convert the 'Time' column to minutes in the month
        # with day info in the column 'Report Date'
        df = DataModule.preprocess_time(df)
        if dat is None:
            dat = df
        else:
            dat = pd.concat([dat, df], ignore_index=True)
    return dat


dat = load_from_xlsx()
# swap Min Delay to the last column
dat = dat[[col for col in dat.columns if col != 'Min Delay'] + ['Min Delay']]
# remove rows that contain empty values
dat = dat.dropna()
# drop the columns 'Station ID'
# dat = dat.drop(columns=['Station ID'])
dat = dat.drop(columns=['Route'])
# convert Min Delay to 0/1 by the limit 30
dat['Min Delay'] = dat['Min Delay'].apply(lambda x: 1 if x > 30 else 0)
dat['Time'] = dat['Time']//120
# dat = dat.drop_duplicates()
# print count of rows in which Min Delay is 0 and 1
# print(dat.drop_duplicates()['Min Delay'].value_counts())
# exit(12)
# split 'dat' to train and test by 0.8 ratio
train = dat.sample(frac=0.8, random_state=998244353)
test = dat.drop(train.index)
r_x_train = train.values[:, :-1]
r_y_train = train.values[:, -1].astype(int)
for i in range(r_x_train.shape[1]):
    r_x_train[:, i] = pd.Categorical(r_x_train[:, i]).codes
r_x_train = pd.get_dummies(pd.DataFrame(r_x_train), columns=[i for i in range(r_x_train.shape[1])]).values
# get the test data
x_test = test.values[:, :-1]
y_test = test.values[:, -1]

# enhance the train data by SMOTEN
smoten = SMOTEN(random_state=998244353)
x_train = train.values[:, :-1]
y_train = train.values[:, -1].astype(int)
# x_train, y_train = smoten.fit_resample(x_train, y_train)
train = np.concatenate([x_train, y_train.reshape((-1, 1))], axis=1)

# save the train and test data
train_dat = pd.DataFrame(columns=dat.columns, data=np.concatenate([x_train, y_train.reshape((-1, 1))], axis=1))
test_dat = pd.DataFrame(columns=dat.columns, data=np.concatenate([x_test, y_test.reshape((-1, 1))], axis=1))

train_dat.to_csv('data/train.csv', index=False)
test_dat.to_csv('data/test.csv', index=False)

# encode all the columns except 'Min Delay'
for col in train_dat.columns:
    if col != 'Min Delay':
        train_dat[col] = train_dat[col].astype('category').cat.codes
        test_dat[col] = test_dat[col].astype('category').cat.codes


train_dat = pd.get_dummies(train_dat, columns=[col for col in train_dat.columns if col != 'Min Delay'])
test_dat = pd.get_dummies(test_dat, columns=[col for col in test_dat.columns if col != 'Min Delay'])
# if the columns in train and test are not the same, add the missing columns to the test data
for col in train_dat.columns:
    if col not in test_dat.columns:
        test_dat[col] = 0
print(f"train cols: {train_dat.columns}")
print(f"test cols: {test_dat.columns}")
print(train_dat.head())

# swap 'Min Delay' to the last column
train_dat = train_dat[[col for col in train_dat.columns if col != 'Min Delay'] + ['Min Delay']]
test_dat = test_dat[[col for col in test_dat.columns if col != 'Min Delay'] + ['Min Delay']]

# get the train and test data
x_train = train_dat.values[:, :-1].astype(float).tolist()
y_train = train_dat.values[:, -1].astype(int).tolist()
x_test = test_dat.values[:, :-1].astype(float).tolist()
y_test = test_dat.values[:, -1].astype(int).tolist()
print(f"feature num: {len(x_train[0])}")
print(f"x_train_len: {len(x_train)}, y_train_len: {len(y_train)}, x_test_len: {len(x_test)}, y_test_len: {len(y_test)}")

# modeling
rf = BalancedRandomForestClassifier(n_estimators=300, criterion='gini', max_depth=None, min_samples_split=16,
                                    max_features='sqrt', n_jobs=-1, random_state=998244353, verbose=1)
# rf = EasyEnsembleClassifier(n_estimators=100, n_jobs=-1, random_state=998244353, verbose=1)
rf.fit(x_train, y_train)
y_hat = rf.predict_proba(x_test)
y_hat = y_hat[:, 1]
balanced_acc = balanced_accuracy_score(y_test, rf.predict(x_test))
print(f"balanced accuracy: {balanced_acc}")
metrics = calculate_metrics(torch.Tensor(y_hat), torch.Tensor(y_test))
print(metrics)
