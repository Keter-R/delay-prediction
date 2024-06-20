import os

import pandas as pd

import utils

stops_path = './data/routes info/stops'
# files with .jpg, .png, .jpeg extensions
stops_img_list = [f for f in os.listdir(stops_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
input_file = './data/ttc-streetcar-delay-data-2020.xlsx'
output_file = './data/ttc-streetcar-delay-data-2020-with-stations.xlsx'
# for i in range(len(stops_img_list)):
#     res_name = utils.extract_stops_name_list_from_img(os.path.join(stops_path, stops_img_list[i]))
#     res_num = utils.extract_stops_number_list_from_img(os.path.join(stops_path, stops_img_list[i]))
#     if len(res_name) > len(res_num):
#         # fill zero to res_num
#         res_num += ['0'] * (len(res_name) - len(res_num))
#     # save csv to stops_path with the same name if file not exists
#     if not os.path.exists(os.path.join(stops_path, stops_img_list[i].split('.')[0] + '.csv')):
#         with open(os.path.join(stops_path, stops_img_list[i].split('.')[0] + '.csv'), 'w', encoding='utf-8') as f:
#             for j in range(len(res_name)):
#                 f.write(res_name[j] + ',' + res_num[j] + '\n')

routes = ['501', '503', '504', '505', '506', '509', '510', '511', '512']
stops = dict()
stops_num = dict()
for route in routes:
    csv_file = os.path.join(stops_path, route + '.csv')
    stations = []
    nums = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        for line in f:
            # if line is empty, skip
            if not line.strip():
                continue
            name, num = line.strip().split(',')
            stations.append(name)
            nums.append(num)
    stops[route] = stations
    stops_num[route] = nums

if input_file is not None and output_file is not None:
    # read each sheet from the Excel file
    xls = pd.ExcelFile(input_file)
    sheet_names = xls.sheet_names
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name in sheet_names:
            df = pd.read_excel(input_file, sheet_name, dtype={'Route': str, 'Line': str, 'Time': str})
            if 'Line' in df.columns:
                df = df.rename(columns={'Line': 'Route'})
            if 'Station' in df.columns:
                df = df.rename(columns={'Station': 'Location'})
            if 'Report Date' in df.columns:
                df = df.rename(columns={'Report Date': 'Date'})

            # for each row, if exists empty columns, remove the row
            df = df.dropna()
            # for each row, if 'Route' is not in routes, remove the row
            df = df[df['Route'].isin(routes)]
            # print row count
            print(df.shape[0])
            # add 'Station ID' column for line
            df['Station ID'] = df['Location']
            # add 'Station' column for line
            df['Station'] = df['Location']
            # set 'Date' format to 'yyyy-mm-dd'
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            # if 'Time' format is 'hh:mm:ss', convert it to 'hh:mm'
            df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%H:%M')
            # try to match the station name with the stops and replace the station name with the stop name
            for row in df.iterrows():
                route = row[1]['Route']
                location = row[1]['Location']
                station = utils.match_station_with_stop(stops[route], location, confidence=80)
                if station is None:
                    print('No station matched for:', location)
                    # remove this row
                    df = df.drop(row[0])
                    continue
                df.at[row[0], 'Station'] = station
                station_id = stops[route].index(station)
                df.at[row[0], 'Station ID'] = stops_num[route][station_id]
            # save the DataFrame to a new Excel file
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print('Sheet:', sheet_name, 'saved to', output_file)

