from sklearn.cluster import KMeans

def train_kmeans(x, k):
    model = KMeans(n_clusters=k, random_state=0)
    y_kmeans = model.fit_predict(x)
    return model, y_kmeans

def elbow_method(x):
    wcss = []
    for i in range(1, 11):
        model = KMeans(n_clusters=i, random_state=0)
        model.fit(x)
        wcss.append(model.inertia_)
    return wcss