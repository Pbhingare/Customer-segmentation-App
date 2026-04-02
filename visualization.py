import matplotlib.pyplot as plt

def plot_clusters(x, y_kmeans, model, k):
    fig, ax = plt.subplots()

    for i in range(k):
        ax.scatter(x[y_kmeans == i, 0],
                   x[y_kmeans == i, 1],
                   label=f'Cluster {i+1}')

    ax.scatter(model.cluster_centers_[:, 0],
               model.cluster_centers_[:, 1],
               s=200, c='black', marker='X',
               label='Centroids')

    ax.set_title("Customer Clusters")
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.legend()

    return fig


def plot_elbow(wcss):
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss)
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("WCSS")
    return fig