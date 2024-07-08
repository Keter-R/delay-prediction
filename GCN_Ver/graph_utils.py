# 3 types of graphs: k_nearest, spatial, temporal
import numpy as np
import pandas as pd
import yaml
import torch
from joblib import Parallel, delayed

threads_count = -1


def prepare_for_gcn(adj, lmb=1.0):
    adj = torch.Tensor(adj).to('cuda')
    adj = adj + lmb * torch.eye(adj.size(0)).to('cuda')
    D = torch.inverse(torch.sqrt(torch.diag(torch.sum(adj, dim=1)))).to('cuda')
    adj = torch.mm(torch.mm(D, adj), D)
    return adj


def generate_spatial_graph(data, route_adj, route_mapper):
    if 'Delay' in data.columns:
        data = data.drop(columns=['Delay'])
    assert 'Station ID' in data.columns
    sid = data['Station ID'].values
    data = data.drop(columns=['Station ID'])
    sp_adj = np.zeros((len(data), len(data)))
    sp_feat = np.zeros((len(data), len(data.columns)))
    for i in range(len(data)):
        sp_feat[i, :] = data.iloc[i].values
    results = Parallel(n_jobs=threads_count)(
        delayed(spatial_adj_parallel)(a, route_adj, route_mapper, sid) for a in range(len(data)))
    sp_adj = np.array(results)
    return sp_adj, sp_feat


def generate_temporal_graph(data, raw_data, time_limit=30):
    if 'Delay' in data.columns:
        data = data.drop(columns=['Delay'])
    if 'Station ID' in data.columns:
        data = data.drop(columns=['Station ID'])
    print(raw_data.columns)
    # in raw_data, join Date and Time to get the time
    raw_data['Time'] = raw_data['Date'] + ' ' + raw_data['Time']
    col_time = pd.to_datetime(raw_data['Time'])
    print(f"raw head: {raw_data.head(5)}")
    te_adj = np.zeros((len(data), len(data)))
    te_feat = np.zeros((len(data), len(data.columns)))
    for i in range(len(data)):
        te_feat[i, :] = data.iloc[i].values
    results = Parallel(n_jobs=threads_count)(
        delayed(temporal_adj_parallel)(a, time_limit, col_time, len(data)) for a in range(len(data)))
    te_adj = np.array(results)
    return te_adj, te_feat


def generate_k_nearest_graph(data, k_neighbors):
    if 'Delay' in data.columns:
        data = data.drop(columns=['Delay'])
    if 'Station ID' in data.columns:
        data = data.drop(columns=['Station ID'])
    ne_adj = np.zeros((len(data), len(data)))
    ne_feat = np.zeros((len(data), len(data.columns)))
    for i in range(len(data)):
        ne_feat[i, :] = data.iloc[i].values
    p = 2
    results = Parallel(n_jobs=threads_count)(
        delayed(neighbor_adj_parallel)(a, k_neighbors, data, len(data), p) for a in range(len(data)))
    ne_adj = np.array(results)
    return ne_adj, ne_feat


def generate_route_graph(data, routes):
    if 'Delay' in data.columns:
        data = data.drop(columns=['Delay'])
    n = len(data)
    ro_adj = np.zeros((len(data), len(data)))
    ro_feat = np.zeros((len(data), len(data.columns)))
    for i in range(0, n):
        ro_feat[i, :] = data.iloc[i].values
    results = Parallel(n_jobs=threads_count)(
        delayed(route_adj_parallel)(a, routes, n) for a in range(n))
    ro_adj = np.array(results)
    return ro_adj, ro_feat


def spatial_adj_parallel(a, route_adj, route_mapper, sid):
    adj = np.zeros(len(sid))
    for b in range(0, len(sid)):
        if b == a:
            continue
        if route_adj[route_mapper[sid[a]], route_mapper[sid[b]]] > 0:
            adj[b] = 1
    return adj


def temporal_adj_parallel(a, time_limit, col_time, n):
    adj = np.zeros(n)
    for b in range(0, len(col_time)):
        if b == a:
            continue
        delt_t = (col_time[a] - col_time[b])
        delt_t = delt_t.total_seconds() / 60
        delt_t = abs(delt_t)
        if delt_t < time_limit:
            adj[b] = 1
    return adj


def neighbor_adj_parallel(a, k_neighbors, data, n, p=2):
    data = data.astype(float)
    adj = np.zeros(n)
    dis = np.zeros(n)
    for b in range(0, n):
        if b == a:
            dis[b] = np.inf
        else:
            dis[b] = np.linalg.norm(data.iloc[a].values - data.iloc[b].values, p)
    # set the k nearest neighbors to 1 in the adj matrix
    for b in range(0, k_neighbors):
        min_dis = np.min(dis)
        if np.isinf(min_dis):
            break
        min_idx = np.argmin(dis)
        adj[min_idx] = 1
        dis[min_idx] = np.inf
    return adj


def route_adj_parallel(a, routes, n):
    adj = np.zeros(n)
    for b in range(0, n):
        if routes[a] == routes[b] and a != b:
            adj[b] = 1
    return adj
