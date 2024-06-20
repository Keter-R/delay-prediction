import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import WeightedRandomSampler


# no shuffle for the DataLoader
class DataModule(pl.LightningDataModule):
    def __init__(self, name, year, batch_size, split_ratio=0.8, seq_len=10, pre_len=1, label_encode=False):
        super(DataModule, self).__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.name = name
        self.year = year
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.sampler = None
        self.label_encode = label_encode
        # if not label_encode, then using sample
        self.using_sample = label_encode
        self.feature_num = 6
        # data format: (month, route, month_minute, day_type, location, incident, delay)
        self.data = self.load_data()
        assert self.data is not None
        self.generate_dataset()

    def setup(self, stage: str = None):
        (
            self.train_dataset,
            self.val_dataset,
        ) = self.generate_dataset()

    def load_data(self):
        if self.name == 'ttc-streetcar-delay-data':
            file_path = 'data/ttc-streetcar-delay-data-{}.xlsx'.format(self.year)
            # read all sheets from the Excel file
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            dat = None
            target_cols = ['Report Date', 'Time', 'Day', 'Incident', 'Min Delay', 'Route', 'Line', 'Date', 'Station ID', 'Delay']
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
                df = self.preprocess_time(self, df)
                if dat is None:
                    dat = df
                else:
                    dat = pd.concat([dat, df], ignore_index=True)
            dat = self.data_encode(self, dat)
            return dat
        return None

    @staticmethod
    def data_encode(self, data):
        for col in ['Day', 'Incident', 'Route']:
            data[col] = data[col].astype('category').cat.codes.astype('int') + 1
        # data = data.drop(columns=['Station ID'])
        # 'Time' column //15
        data['Time'] = data['Time'] // 15

        data = pd.get_dummies(data, columns=['Time'])
        data = pd.get_dummies(data, columns=['Day'])
        data = pd.get_dummies(data, columns=['Incident'])
        data = pd.get_dummies(data, columns=['Month'])
        data = pd.get_dummies(data, columns=['Route'])
        # change the column 'Min Delay' to 0/1 by the limit 30
        if self.label_encode:
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
        cols.remove('Station ID')
        cols.append('Station ID')
        cols.append('Min Delay')
        data = data[cols]

        print(data.columns)
        print(data.head())
        return data

    @staticmethod
    def preprocess_time(self, df):
        # convert the 'Time' column to minutes in the month
        # with day info in the column 'Report Date'
        for i in range(len(df)):
            # cast str(00:00) to datetime.time
            if len(str(df.iloc[i, 2])) == 5:
                df.iloc[i, 2] = pd.to_datetime(df.iloc[i, 2], format='%H:%M').time()
            time = df.iloc[i, 2]
            day = pd.to_datetime(df.iloc[i, 0])
            day_min = (day.day - 1) * 24 * 60
            month_min = day_min + time.hour * 60 + time.minute
            # normalize
            df.iloc[i, 2] = month_min - day_min
        # set the column 'Report Date' to 'Month'
        cols = list(df.columns)
        if 'Report Date' in cols:
            df = df.rename(columns={'Report Date': 'Month'})
        elif 'Date' in cols:
            df = df.rename(columns={'Date': 'Month'})
        # set the column 'Month' to int type
        df['Month'] = pd.to_datetime(df['Month']).dt.month
        return df

    def generate_batch_data(self, data, using_sample=False):
        data = data.values
        data_len = len(data)
        # check if str in data
        # for i in range(data_len):
        #     for j in range(len(data[i])):
        #         if type(data[i][j]) is str:
        #             print(data[i])
        # exit(0)
        x, y = list(), list()
        for i in range(data_len - self.seq_len - self.pre_len + 1):
            x.append(data[i:i + self.seq_len, :])
            y.append(data[i + self.seq_len - 1 + self.pre_len, -1])
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
        dataset = torch.utils.data.TensorDataset(x, y)
        return dataset

    def generate_dataset(self):
        all_dataset = self.generate_batch_data(self.data)
        train_size = int(len(all_dataset) * self.split_ratio)
        test_size = len(all_dataset) - train_size
        print(len(all_dataset))
        print(train_size, test_size)
        train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
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
            self.sampler = WeightedRandomSampler(samples_weight, train_size)
        return train_dataset, test_dataset

    def train_dataloader(self):
        if self.sampler is not None:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=0)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0)
        # return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), num_workers=0)
