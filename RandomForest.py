import torch
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

from ModelModule import calculate_metrics


def get_data_from_dataloader_with_graph(dataloader, seq_len, feature_num, batch_size):
    x = []
    y = []
    for batch in dataloader:
        x_batch, y_batch = batch
        x_batch = x_batch[:, -feature_num:-1]
        x.extend(x_batch)
        y.extend(y_batch)
    return x, y


class RandomForest:
    def __init__(self, data_module, seq_len, batch_size, feature_num, graph_feature_num=0, node_num=0, graph=True,
                 n_classes=2,
                 n_estimators=1500,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=16,
                 max_features='sqrt',
                 n_jobs=-1,
                 random_state=998244353,
                 verbose=1,
                 class_weight="balanced"
                 ):
        self.n_classes = n_classes
        # self.clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
        #                                   min_samples_split=min_samples_split, max_features=max_features,
        #                                   n_jobs=n_jobs, random_state=random_state, verbose=verbose,
        #                                   class_weight=class_weight)
        self.clf = BalancedRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                  min_samples_split=min_samples_split, max_features=max_features,
                                                  n_jobs=n_jobs, random_state=random_state, verbose=verbose,
                                                  replacement=True, bootstrap=False)
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.get_data(data_module, seq_len, batch_size, feature_num, graph_feature_num, node_num, using_graph=graph)

    def fit(self):
        self.clf.fit(self.x_train, self.y_train)
        # accuracy = cross_val_score(self.clf, self.x_train, self.y_train, scoring='accuracy', cv=5)
        # precision = cross_val_score(self.clf, self.x_train, self.y_train, scoring='precision', cv=5)
        # recall = cross_val_score(self.clf, self.x_train, self.y_train, scoring='recall', cv=5)
        # f1_score = cross_val_score(self.clf, self.x_train, self.y_train, scoring='f1', cv=5)
        # auc = cross_val_score(self.clf, self.x_train, self.y_train, scoring='roc_auc', cv=5)
        # print("准确率:", accuracy.mean())
        # print("精确率:", precision.mean())
        # print("召回率:", recall.mean())
        # print("F1_score:", f1_score.mean())
        # print("AUC:", auc.mean())

    def validate(self):
        y_hat = self.clf.predict_proba(self.x_val)
        y_hat = y_hat[:, 1]
        print(y_hat)
        print("Validation Report:")
        metrics = calculate_metrics(torch.Tensor(y_hat), torch.Tensor(self.y_val))
        print(metrics)

    def predict(self, x):
        return self.clf.predict(x)

    def get_data(self, data_module, seq_len, batch_size, feature_num, graph_feature_num, node_num, using_graph=True):
        if using_graph:
            print("Using graph data")
            train_dataloader = data_module.train_dataloader()
            val_dataloader = data_module.val_dataloader()
            self.x_train, self.y_train = get_data_from_dataloader_with_graph(train_dataloader, seq_len, feature_num,
                                                                             batch_size)
            self.x_val, self.y_val = get_data_from_dataloader_with_graph(val_dataloader, seq_len, feature_num,
                                                                         batch_size)
        else:
            print("Using station data")
            raise NotImplementedError
