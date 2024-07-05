import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import graph_utils


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path,
                 raw_cols, graph_feature_cols,
                 node_num=343, k_neighbors=3,
                 split_ratio=0.8, delt_t=20, seed=0, time_limit=30):
        super(DataModule, self).__init__()
        self.np_ratio = -1
        self.graph_feature_num = -1
        self.feature_num = -1
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_index = []
        self.val_index = []
        self.split_ratio = split_ratio
        self.delt_t = delt_t
        self.seed = seed
        self.time_limit = time_limit
        self.node_num = node_num
        self.k_neighbors = k_neighbors
        self.raw_cols = raw_cols
        self.graph_feature_cols = graph_feature_cols
        self.raw_data = None
        self.data_len = 0
        self.data = self.load_data(data_path)
        self.route_adj, self.route_mapper = self.load_adj()
        self.sample(self.data)
        self.generate_dataset(self.data, self.train_index, self.val_index)

    def setup(self, stage: str = None):
        print(f"setup stage: {stage}")

    def load_data(self, file_path):
        # read all sheets from the Excel file
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        dat = None
        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name)
            # remove cols that not in target_cols
            for col in df.columns:
                if col not in self.raw_cols:
                    df = df.drop(columns=[col])
            if dat is None:
                dat = df
            else:
                dat = pd.concat([dat, df], ignore_index=True)
        # convert 'Delay' to 0/1 by the limit 30 if Delay >= 30
        if dat['Delay'].max() > 1:
            dat['Delay'] = (dat['Delay'] >= 30).astype(int)
        pos = 0
        neg = 0
        for i in range(len(dat)):
            if dat['Delay'][i] == 1:
                pos += 1
            else:
                neg += 1
        print(f"positive count: {pos}")
        print(f"negative count: {neg}")
        print(f"raw data row count: {len(dat)}")
        self.raw_data = dat
        dat = self.preprocess_time(dat)
        dat['Time'] = dat['Time'] // self.delt_t
        dat = pd.get_dummies(dat, columns=self.graph_feature_cols)
        self.graph_feature_num = len(dat.columns) - 2
        self.feature_num = len(dat.columns) - 2
        self.data_len = len(dat)
        print(f"graph feature num: {self.graph_feature_num}")
        print(dat.columns)
        return dat

    def sample(self, data):
        sampled_data = [i for i in range(0, len(data))]
        random.seed(self.seed)
        random.shuffle(sampled_data)
        self.train_index = sampled_data[:int(len(sampled_data) * self.split_ratio)]
        self.val_index = sampled_data[int(len(sampled_data) * self.split_ratio):]

    def generate_dataset(self, data, train_index, val_index):
        self.generate_traditional_data(data, train_index, val_index)
        spatial_adj, spatial_feature = graph_utils.generate_spatial_graph(data, self.route_adj, self.route_mapper)
        temporal_adj, temporal_feature = graph_utils.generate_temporal_graph(data, self.raw_data, self.time_limit)
        neighbor_adj, neighbor_feature = graph_utils.generate_k_nearest_graph(data, self.k_neighbors)
        x = np.concatenate([spatial_adj.flatten(), spatial_feature.flatten(), temporal_adj.flatten(),
                            temporal_feature.flatten(), neighbor_adj.flatten(), neighbor_feature.flatten()])
        y_train = [self.train_index, self.y_train]
        y_val = [self.val_index, self.y_val]
        x = torch.tensor(np.array([x]), dtype=torch.float32)
        y_train = torch.tensor(np.array([y_train]), dtype=torch.float32)
        y_val = torch.tensor(np.array([y_val]), dtype=torch.float32)
        self.train_dataset = torch.utils.data.TensorDataset(x, y_train)
        self.val_dataset = torch.utils.data.TensorDataset(x, y_val)

    def generate_traditional_data(self, data, train_index, val_index):
        if 'Station ID' in data.columns:
            data = data.drop(columns=['Station ID'])
        if 'Delay' in data.columns:
            data_cols = list(data.columns)
            data_cols.remove('Delay')
            data = data[data_cols + ['Delay']]
        print(data.head(5))
        self.x_train = data.iloc[train_index, :-1].values
        self.y_train = data.iloc[train_index, -1].values
        self.x_val = data.iloc[val_index, :-1].values
        self.y_val = data.iloc[val_index, -1].values
        self.np_ratio = (len(self.y_train) - self.y_train.sum()) / self.y_train.sum()

    def load_adj(self):
        routes = ['501', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512']
        file_path = 'data/routes info/stops/'
        node_num = 0
        node_mapper = dict()
        for route in routes:
            csv_file = file_path + route + '.csv'
            with open(csv_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    name, num = line.strip().split(',')
                    # convert num to int
                    num = int(num)
                    if num not in node_mapper:
                        node_mapper[num] = node_num
                        node_num += 1
        adj_mat = np.zeros((node_num, node_num))
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
                adj_mat[node_mapper[nums[i]], node_mapper[nums[i + 1]]] = 1
                adj_mat[node_mapper[nums[i + 1]], node_mapper[nums[i]]] = 1
        return adj_mat, node_mapper

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

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=0)
