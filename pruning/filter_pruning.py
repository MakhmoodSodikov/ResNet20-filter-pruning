from sklearn.cluster import KMeans
from model.resnet20 import ResNet20
num_clusters = 3
# https://arxiv.org/pdf/1806.09228.pdf
# https://github.com/VITA-Group/Deep-K-Means-pytorch/blob/master/util/kmeans.py
# https://medium.com/depurr/pytorch-inter-epoch-training-with-checkpoints-bac8477828d
model = ResNet20()

weights = model.layer1[0].conv1.weight.detach().numpy()
weights = weights.reshape(len(weights), -1)
print(weights.shape)
print(type(weights))

kmeans = KMeans(
    init='k-means++',
    n_clusters=num_clusters,
    n_init=10)

kmeans.fit(weights)

print(kmeans.cluster_centers_.reshape(3, 16, 3, 3))
print(kmeans.labels_)
