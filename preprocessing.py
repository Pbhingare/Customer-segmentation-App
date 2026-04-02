def get_features(dataset):
    return dataset.iloc[:, [3, 4]].values