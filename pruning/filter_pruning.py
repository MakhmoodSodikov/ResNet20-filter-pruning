from sklearn.cluster import KMeans
from model.resnet20 import ResNet20
import torch


# https://arxiv.org/pdf/1806.09228.pdf
# https://medium.com/depurr/pytorch-inter-epoch-training-with-checkpoints-bac8477828d


def filter_pruning_20(model_path, num_clusters=16):
    """
    Naive filter pruning implementation
    :model_path: Path to torch .th-file with model pretrained weights
    :return: model with clustered conv layers
    """
    dump = torch.load(model_path)
    model = ResNet20()
    model.load_state_dict(dump['state_dict'])

    # Local constants
    LAYERS = {'layer1': model.layer1, 'layer2': model.layer2, 'layer3': model.layer3}
    CONVS = ['conv1', 'conv2']
    BLOCK_COUNT = 3

    # First conv layer
    model.conv1.weight = clusterize(model.conv1.weight, num_clusters)
    for layer in LAYERS:
        for block in range(BLOCK_COUNT):
            for conv_name in CONVS:
                if conv_name == 'conv1':
                    weights = LAYERS[layer][block].conv1.weight
                    LAYERS[layer][block].conv1.weight = clusterize(weights, num_clusters)
                else:
                    weights = LAYERS[layer][block].conv2.weight
                    LAYERS[layer][block].conv2.weight = clusterize(weights, num_clusters)
    return model


def fix_num_clusters_test(model_path, num_clusters_dict):
    assert 'layer1' in num_clusters_dict and 'layer2' in num_clusters_dict and \
           'layer3' in num_clusters_dict and 'layer0' in num_clusters_dict, \
        'Num_clusters for each layer are required!'

    for key, val in num_clusters_dict.items():
        if key != 'layer0':
            num_clusters_dict[key] = [val for _ in range(6)]

    return flex_num_clusters_test(model_path, num_clusters_dict)


def flex_num_clusters_test(model_path, num_clusters_dict):
    assert 'layer1' in num_clusters_dict and 'layer2' in num_clusters_dict and \
           'layer3' in num_clusters_dict and 'layer0' in num_clusters_dict, \
        'Num_clusters for each layer are required!'

    dump = torch.load(model_path)
    model = ResNet20()
    model.load_state_dict(dump['state_dict'])

    # Local constants
    LAYERS = {'layer1': model.layer1, 'layer2': model.layer2, 'layer3': model.layer3}
    CONVS = ['conv1', 'conv2']
    BLOCK_COUNT = 3

    # First conv layer
    model.conv1.weight = clusterize(model.conv1.weight, num_clusters_dict['layer0'])
    for layer in LAYERS:
        num_clusters_layer = num_clusters_dict[layer]
        conv_counter = 0
        for block in range(BLOCK_COUNT):

            for conv_name in CONVS:

                if conv_name == 'conv1':
                    weights = LAYERS[layer][block].conv1.weight
                    LAYERS[layer][block].conv1.weight = clusterize(weights, num_clusters_layer[conv_counter])
                else:
                    weights = LAYERS[layer][block].conv2.weight
                    LAYERS[layer][block].conv2.weight = clusterize(weights, num_clusters_layer[conv_counter])
                conv_counter += 1
    return model


def clusterize(weights, num_clusters):
    weights = weights.detach().numpy()
    kmeans = KMeans(init='k-means++',
                    n_clusters=num_clusters,
                    n_init=10)
    kernel_size = weights.shape[2:]
    init_shape = weights.shape
    weights = weights.reshape(-1, kernel_size[0] * kernel_size[1])

    kmeans.fit(weights)
    for i, label in enumerate(kmeans.labels_):
        weights[i] = kmeans.cluster_centers_[label]
    return torch.nn.Parameter(torch.Tensor(weights.reshape(init_shape)), requires_grad=True)
