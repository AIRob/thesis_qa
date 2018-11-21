from sklearn.cluster import KMeans,AgglomerativeClustering
def kmeans(vectors,k):
    kmeans = KMeans(n_clusters=k).fit(vectors)
    return  kmeans.labels_

def hierachical_cluster(vectors,k):
    tmp = AgglomerativeClustering(k).fit(vectors)
    return tmp.labels_ 
def cluster_by_labels(items,labels,k):
    assert len(items) == len(labels)
    cluster_list =  [[] for _ in range(k)]
    for item,label in zip(items,labels):
        cluster_list[label].append(item)
    return cluster_list





