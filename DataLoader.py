import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import WeightedRandomSampler
from imblearn.over_sampling import SMOTEN


def resample(dat, n):
    # swap 'Min Delay' to the last column
    cols = list(dat.columns)
    cols.remove('Min Delay')
    cols.append('Min Delay')
    dat = dat[cols]
    data = dat.values
    # drop row that contains empty value
    data = data[~pd.isnull(data).any(axis=1)]
    print(f"resample data with n={n}")
    # get dataframe of x
    x = data[:, :-1]
    y = data[:, -1]
    # if string in x columns, then
    # if y is not 0/1, then convert it to 0/1
    if y.max() > 1:
        y = (y >= 30).astype(int)
    print(x)
    print(y)
    smote_n = SMOTEN(random_state=42)
    x_res, y_res = smote_n.fit_resample(x, y)
    data_ = np.concatenate((x_res, y_res.reshape(-1, 1)), axis=1)
    print(f"resample data shape: {data_.shape}")
    dat_ = pd.DataFrame(data=data_, columns=dat.columns)
    dat_.to_csv("data/temp.csv", index=False)
    # exit(1770)
    return dat_


# no shuffle for the DataLoader
class DataModule(pl.LightningDataModule):
    def __init__(self, name, year, batch_size, split_ratio=0.8, seq_len=10, label_encode=False,
                 with_station_id=False, delt_t=360, graph=False):
        super(DataModule, self).__init__()
        self.np_ratio = 10
        self.val_dataset = None
        self.train_dataset = None
        self.val_dataldr = None
        self.train_dataldr = None
        self.name = name
        self.year = year
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.with_station_id = with_station_id
        self.seq_len = seq_len
        self.sampler = None
        self.graph_feature_num = 0
        self.graph = graph
        self.delt_t = delt_t
        self.intervals = 24 * 60 // delt_t
        self.label_encode = label_encode
        # if not label_encode, then using sample
        self.using_sample = label_encode
        self.feature_num = 6
        self.node_num = 0
        self.adj_mat = None
        self.node_mapper = None
        # data format: (month, route, month_minute, day_type, location, incident, delay)
        self.raw_data = None
        self.data = self.load_data()
        assert self.data is not None
        self.train_dataset, self.val_dataset = self.generate_dataset()

    def setup(self, stage: str = None):
        print(f"setup stage: {stage}")

    def load_data(self):
        if self.name == 'ttc-streetcar-delay-data':
            file_path = 'data/ttc-streetcar-delay-data-{}.xlsx'.format(self.year)
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
                df = self.preprocess_time(df)
                if dat is None:
                    dat = df
                else:
                    dat = pd.concat([dat, df], ignore_index=True)
            # dat = resample(dat, 2 * len(dat))
            dat = self.data_encode(dat)
            return dat
        return None

    def load_adj(self):
        routes = ['501', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512']
        file_path = 'data/routes info/stops/'
        self.node_num = 0
        self.node_mapper = dict()
        for route in routes:
            csv_file = file_path + route + '.csv'
            with open(csv_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    name, num = line.strip().split(',')
                    # convert num to int
                    num = int(num)
                    if num not in self.node_mapper:
                        self.node_mapper[num] = self.node_num
                        self.node_num += 1
        self.adj_mat = np.zeros((self.node_num, self.node_num))
        # conjunct the stops in the same route
        for route in routes:
            csv_file = file_path + route + '.csv'
            stations = []
            nums = []
            with open(csv_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    name, num = line.strip().split(',')
                    stations.append(name)
                    nums.append(int(num))
            for i in range(len(nums) - 1):
                self.adj_mat[self.node_mapper[nums[i]], self.node_mapper[nums[i + 1]]] = 1
                self.adj_mat[self.node_mapper[nums[i + 1]], self.node_mapper[nums[i]]] = 1

    def node_map(self, data):
        data['Station ID'] = data['Station ID'].map(self.node_mapper)
        assert data['Station ID'].isnull().sum() == 0
        return data

    def data_encode(self, data):
        # for col in ['Day', 'Incident', 'Route']:
        #     if col not in data.columns:
        #         print(f"column {col} not in the data")
        #         continue
        #     data[col] = data[col].astype('category').cat.codes.astype('int') + 1
        if not self.with_station_id:
            data = data.drop(columns=['Station ID'])
        else:
            self.load_adj()
            data = self.node_map(data)
        # data = pd.get_dummies(data, columns=['Station ID'])
        # 'Time' column //15
        data['Time'] = data['Time'] // self.delt_t
        self.raw_data = data
        data = pd.get_dummies(data, columns=['Time'])
        data = pd.get_dummies(data, columns=['Day'])
        data = pd.get_dummies(data, columns=['Incident'])
        data = pd.get_dummies(data, columns=['Month'])
        data = pd.get_dummies(data, columns=['Route'])
        # change the column 'Min Delay' to 0/1 by the limit 30 if Min Delay >= 30
        if self.label_encode and data['Min Delay'].max() > 1:
            data['Min Delay'] = (data['Min Delay'] >= 30).astype(int)
        else:
            data['Min Delay'] = data['Min Delay']  # / 30
        # swap 'Min Delay' to the last column
        cols = list(data.columns)
        cols.remove('Min Delay')
        cols.append('Min Delay')
        data = data[cols]
        self.feature_num = len(data.columns) - 1
        # move 'Station ID' to the second column from the bottom
        cols = list(data.columns)
        cols.remove('Min Delay')
        if 'Station ID' in cols:
            cols.remove('Station ID')
            cols.append('Station ID')
        cols.append('Min Delay')
        data = data[cols]

        print(data.columns)
        print(data.head())
        return data

    @staticmethod
    def preprocess_time(df):
        # find the column 'Time'
        cols = list(df.columns)
        if 'Report Date' in cols:
            df = df.rename(columns={'Report Date': 'Month'})
        elif 'Date' in cols:
            df = df.rename(columns={'Date': 'Month'})

        # # set the 'Day' column to the day of the month
        # if 'Day' in cols:
        #     df['Day'] = pd.to_datetime(df['Month']).dt.day
        df['Month'] = pd.to_datetime(df['Month']).dt.month
        # convert time to 'HH:MM:SS' if the time is in 'HH:MM'
        if 'Time' in cols:
            df['Time'] = df['Time'].apply(lambda x: str(x) + ':00' if len(str(x)) == 5 else x)
            df['Time'] = (pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour * 60
                          + pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute)
        return df

    def generate_batch_data(self, data, using_sample=False):
        r_data = data
        data = data.values
        data_len = len(data)
        # check if str in data
        # for i in range(data_len):
        #     for j in range(len(data[i])):
        #         if type(data[i][j]) is str:
        #             print(data[i])
        # exit(0)

        if not self.graph:
            x, y = list(), list()
            for i in range(data_len - self.seq_len):
                x.append(data[i:i + self.seq_len, :])
                y.append(data[i + self.seq_len - 1, -1])
                # mask the last day's delay info to -1
                x[-1][-1, -1] = -1
            # conversion path: list -> numpy -> tensor (float32) 'maybe faster'
            #x = np.array(x)
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            # x = x[0:400]
            # y = torch.tensor(y[0:400], dtype=torch.float32)
            # check all nan in x and y, fill nan with 0
            if torch.isnan(x).sum() > 0:
                x[torch.isnan(x)] = 0
            if torch.isnan(y).sum() > 0:
                y[torch.isnan(y)] = 0
        else:
            x, y = list(), list()
            cols = list(r_data.columns)
            r_cols_id = []
            use_min_delay = False
            unrelated_cols = ['Incident_Investigation', 'Incident_Emergency Services', 'Incident_Overhead - Pantograph',
                              'Incident_General Delay', 'Route']
            # unrelated_cols = []
            for col in cols:
                # if values in unrelated_cols is substring of col or col is substring of values in unrelated_cols
                # then continue
                if any([uc in col for uc in unrelated_cols]):
                    continue
                if 'Incident' in col:
                    r_cols_id.append(cols.index(col))
                    print('add col:', col)
                if 'Route' in col:
                    r_cols_id.append(cols.index(col))
                    print('add col:', col)
                # if 'Min Delay' in col:
                # r_cols_id.append(cols.index(col))
                # print('add col:', col)
                # use_min_delay = True
                if 'Time' in col:
                    r_cols_id.append(cols.index(col))
                    print('add col:', col)
                if 'Month' in col:
                    r_cols_id.append(cols.index(col))
                    print('add col:', col)
                if 'Day' in col:
                    r_cols_id.append(cols.index(col))
                    print('add col:', col)
            self.graph_feature_num = len(r_cols_id)
            for i in range(data_len - self.seq_len):
                xin = np.zeros((self.seq_len, self.node_num, self.graph_feature_num))
                for j in range(self.seq_len):
                    if j > 0:
                        xin[j, :] = xin[j - 1, :, :]
                    sid = int(data[i + j, -2])
                    xin[j, sid, :] = data[i + j, r_cols_id]
                    # mask the last day's delay info to -1
                    if j == self.seq_len - 1 and use_min_delay:
                        xin[j, sid, -1] = -1
                xfeat = data[i + self.seq_len - 1, :-1]
                # squeeze the xin to 1D tensor
                xin = xin.flatten()
                x.append(np.concatenate((xin, xfeat), axis=0).astype(np.float32))
                y.append(data[i + self.seq_len - 1, -1])
            x = torch.tensor(np.array(x), dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x, y)
        return dataset

    def generate_dataset(self):
        all_dataset = self.generate_batch_data(self.data)
        train_size = int(len(all_dataset) * self.split_ratio)
        test_size = len(all_dataset) - train_size
        print(len(all_dataset))
        print(train_size, test_size)
        train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
        if self.using_sample:
            # calculate the sample weight
            y = train_dataset
            positive_count = 0
            negative_count = 0
            for (a, b) in y:
                if b == 1:
                    positive_count += 1
                else:
                    negative_count += 1
            samples_weight = []
            for (a, b) in y:
                if b == 1:
                    samples_weight.append(1 / positive_count)
                else:
                    samples_weight.append(1 / negative_count)
            print('Train P count:', positive_count)
            print('Train N count:', negative_count)
            print('Train N/P ratio:', negative_count / positive_count)
            self.sampler = WeightedRandomSampler(samples_weight, positive_count + negative_count, replacement=True)
            self.sampler = None
            self.np_ratio = negative_count / positive_count
        return train_dataset, val_dataset

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_dataloader(self):
        if self.sampler is not None:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=0)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)
