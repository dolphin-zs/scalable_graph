import torch
import math
import numpy as np
import torch.distributed as dist

from torch_geometric.data import Data, ClusterData, ClusterLoader, NeighborSampler
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from graph_saint import MySAINTRandomWalkSampler


class NeighborSampleDataset(IterableDataset):
    def __init__(self, X, y, edge_index, edge_weight, num_nodes, batch_size,
                 shuffle=False, use_dist_sampler=False, rep_eval=None,
                 neighbor_sampling_size=5, neighbor_batch_size=100):
        self.X = X
        self.y = y

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        # whether to use distributed sampler when available
        self.use_dist_sampler = use_dist_sampler
        # the number of repeats to run evaluation, set to None for training mode
        self.rep_eval = rep_eval
        # the number of neighbors to be sampled at each layer
        self.neighbor_sampling_size = neighbor_sampling_size
        # the number of central nodes to be expanded at each batch
        self.neighbor_batch_size = neighbor_batch_size

        # use 'epoch' as the random seed to shuffle data for distributed training
        self.epoch = None

        self.graph_sampler = self._make_graph_sampler()
        self.length = self.get_length()

    def _make_graph_sampler(self):
        graph = Data(edge_index=self.edge_index,
                     edge_attr=self.edge_weight,
                     num_nodes=self.num_nodes)

        # TODO: refactor to fit arbitrary number of layers
        graph_sampler = NeighborSampler(graph,
                                        size=[self.neighbor_sampling_size]*2,
                                        num_hops=2,
                                        batch_size=self.neighbor_batch_size,
                                        shuffle=self.shuffle,
                                        add_self_loops=True)

        return graph_sampler

    def get_subgraph(self, data_flow):
        sub_graph = {
            'type': 'dataflow',
            'edge_index': [block.edge_index for block in data_flow],
            'edge_weight': [self.edge_weight[block.e_id] for block in data_flow],
            'size': [block.size for block in data_flow],
            'n_id': [block.n_id for block in data_flow],
            'res_n_id': [block.res_n_id for block in data_flow],
            'cent_n_id': data_flow[-1].n_id[data_flow[-1].res_n_id],
            'graph_n_id': data_flow[0].n_id
        }

        return sub_graph

    def __iter__(self):
        repeats = 1 if self.rep_eval is None else self.rep_eval

        for rep in range(repeats):
            # decide random seeds for graph sampler

            if self.use_dist_sampler and dist.is_initialized():
                # ensure that all processes share the same graph dataflow
                # set seed as epoch for training, and rep for evaluation
                torch.manual_seed(self.epoch)

            if self.rep_eval is not None:
                # fix random seeds for repetitive evaluation
                # this attribute should not be set during training
                torch.manual_seed(rep)

            for data_flow in self.graph_sampler():
                g = self.get_subgraph(data_flow)
                X, y = self.X[:, g['graph_n_id']], self.y[:, g['cent_n_id']]
                dataset_len = X.size(0)
                indices = list(range(dataset_len))

                if self.use_dist_sampler and dist.is_initialized():
                    # distributed sampler reference: torch.utils.data.distributed.DistributedSampler
                    if self.shuffle:
                        # ensure that all processes share the same permutated indices
                        tg = torch.Generator()
                        tg.manual_seed(self.epoch)
                        indices = torch.randperm(
                            dataset_len, generator=tg).tolist()

                    world_size = dist.get_world_size()
                    node_rank = dist.get_rank()
                    num_samples_per_node = int(
                        math.ceil(dataset_len * 1.0 / world_size))
                    total_size = world_size * num_samples_per_node

                    # add extra samples to make it evenly divisible
                    indices += indices[:(total_size - dataset_len)]
                    assert len(indices) == total_size

                    # get sub-batch for each process
                    # Node (rank=x) get [x, x+world_size, x+2*world_size, ...]
                    indices = indices[node_rank:total_size:world_size]
                    assert len(indices) == num_samples_per_node
                elif self.shuffle:
                    np.random.shuffle(indices)

                num_batches = (len(indices) + self.batch_size -
                            1) // self.batch_size
                for batch_id in range(num_batches):
                    start = batch_id * self.batch_size
                    end = (batch_id + 1) * self.batch_size
                    yield X[indices[start: end]], y[indices[start: end]], g, torch.LongTensor(indices[start: end])

    def get_length(self):
        length = 0

        repeats = 1 if self.rep_eval is None else self.rep_eval

        for rep in range(repeats):
            for data_flow in self.graph_sampler():
                if self.use_dist_sampler and dist.is_initialized():
                    dataset_len = self.X.size(0)
                    world_size = dist.get_world_size()
                    num_samples_per_node = int(
                        math.ceil(dataset_len * 1.0 / world_size))
                else:
                    num_samples_per_node = self.X.size(0)
                length += (num_samples_per_node +
                        self.batch_size - 1) // self.batch_size

        return length

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        # self.set_epoch() will be called by BasePytorchTask on each epoch when using distributed training
        self.epoch = epoch


class ClusterDataset(IterableDataset):
    def __init__(self, X, y, edge_index, edge_weight, num_nodes, batch_size,
                 shuffle=False, use_dist_sampler=False, rep_eval=None,
                 cluster_part_num=100, cluster_batch_size=10):
        self.X = X
        self.y = y

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        # whether to use distributed sampler when available
        self.use_dist_sampler = use_dist_sampler
        # number of repeats to run evaluation, set to None for training
        self.rep_eval = rep_eval
        # the number of partitions to divide the graph
        self.cluster_part_num = cluster_part_num
        # the number of partitions to load at each batch
        self.cluster_batch_size = cluster_batch_size

        # use 'epoch' as the random seed to shuffle data for distributed training
        self.epoch = None

        self.graph_sampler = self._make_graph_sampler()
        self.length = self.get_length()

    def _make_graph_sampler(self):
        graph = Data(edge_index=self.edge_index,
                     edge_attr=self.edge_weight,
                     num_nodes=self.num_nodes,
                     n_id=torch.arange(self.num_nodes))

        cluster_data = ClusterData(graph,
                                   num_parts=self.cluster_part_num)

        cluster_loader = ClusterLoader(cluster_data,
                                       batch_size=self.cluster_batch_size,
                                       shuffle=True)

        return cluster_loader

    def get_subgraph(self, cluster_part):
        graph = {
            'type': 'subgraph',
            'edge_index': cluster_part.edge_index,
            'edge_weight': cluster_part.edge_attr,
            'cent_n_id': cluster_part.n_id,
        }

        return graph

    def __iter__(self):
        repeats = 1 if self.rep_eval is None else self.rep_eval

        for rep in range(repeats):
            if self.use_dist_sampler and dist.is_initialized():
                # ensure that all processes share the same graph dataflow
                # set seed as epoch for training, and rep for evaluation
                torch.manual_seed(self.epoch)

            if self.rep_eval is not None:
                # fix random seeds for repetitive evaluation
                # this attribute should not be set during training
                torch.manual_seed(rep)

            for cluster_part in self.graph_sampler:
                g = self.get_subgraph(cluster_part)
                X, y = self.X[:, g['cent_n_id']], self.y[:, g['cent_n_id']]
                dataset_len = X.size(0)
                indices = list(range(dataset_len))

                if self.use_dist_sampler and dist.is_initialized():
                    # distributed sampler reference: torch.utils.data.distributed.DistributedSampler
                    if self.shuffle:
                        # ensure that all processes share the same permutated indices
                        tg = torch.Generator()
                        tg.manual_seed(self.epoch)
                        indices = torch.randperm(dataset_len, generator=tg).tolist()

                    world_size = dist.get_world_size()
                    node_rank = dist.get_rank()
                    num_samples_per_node = int(math.ceil(dataset_len * 1.0 / world_size))
                    total_size = world_size * num_samples_per_node

                    # add extra samples to make it evenly divisible
                    indices += indices[:(total_size - dataset_len)]
                    assert len(indices) == total_size

                    # get sub-batch for each process
                    # Node (rank=x) get [x, x+world_size, x+2*world_size, ...]
                    indices = indices[node_rank:total_size:world_size]
                    assert len(indices) == num_samples_per_node
                elif self.shuffle:
                    np.random.shuffle(indices)

                num_batches = (len(indices) + self.batch_size - 1) // self.batch_size
                for batch_id in range(num_batches):
                    start = batch_id * self.batch_size
                    end = (batch_id + 1) * self.batch_size
                    yield X[indices[start: end]], y[indices[start: end]], g, torch.LongTensor(indices[start: end])

    def get_length(self):
        length = 0
        for _ in self.graph_sampler:
            if self.use_dist_sampler and dist.is_initialized():
                dataset_len = self.X.size(0)
                world_size = dist.get_world_size()
                num_samples_per_node = int(math.ceil(dataset_len * 1.0 / world_size))
            else:
                num_samples_per_node = self.X.size(0)
            length += (num_samples_per_node + self.batch_size - 1) // self.batch_size

        return length

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        # self.set_epoch() will be called by BasePytorchTask on each epoch when using distributed training
        self.epoch = epoch


class SAINTDataset(IterableDataset):
    def __init__(self, X, y, edge_index, edge_weight, num_nodes, batch_size,
                 shuffle=False, use_dist_sampler=False, rep_eval=None,
                 saint_batch_size=50, saint_walk_length=2, saint_sample_coverage=50):
        self.X = X
        self.y = y

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        # whether to use distributed sampler when available
        self.use_dist_sampler = use_dist_sampler
        # number of repeats to run evaluation, set to None for training
        self.rep_eval = rep_eval
        # the number of initial nodes for random walk at each batch
        self.saint_batch_size = saint_batch_size
        # the length of each random walk
        self.saint_walk_length = saint_walk_length
        # the number of samples per node used to compute normalization statistics
        self.saint_sample_coverage = saint_sample_coverage

        # use 'epoch' as the random seed to shuffle data for distributed training
        self._epoch = None

        self._graph_sampler = self._make_graph_sampler()
        self._length = self.get_length()

    def _make_graph_sampler(self):
        graph = Data(edge_index=self.edge_index,
                     edge_attr=self.edge_weight,
                     num_nodes=self.num_nodes)

        saint_sampler = MySAINTRandomWalkSampler(
            graph, self.saint_batch_size, self.saint_walk_length,
            sample_coverage=self.saint_sample_coverage, save_dir=None, log=False)

        return saint_sampler

    def get_subgraph(self, saint_data):
        graph = {
            'type': 'subgraph',
            'edge_index': saint_data.edge_index,
            'edge_weight': saint_data.edge_attr,
            'cent_n_id': saint_data.n_id,
            'res_n_id': saint_data.res_n_id,
            'node_norm': saint_data.node_norm,
            'edge_norm': saint_data.edge_norm,
        }

        return graph

    def __iter__(self):
        repeats = 1 if self.rep_eval is None else self.rep_eval

        for rep in range(repeats):
            if self.use_dist_sampler and dist.is_initialized():
                # ensure that all processes share the same graph dataflow
                # set seed as epoch for training, and rep for evaluation
                torch.manual_seed(self._epoch)

            if self.rep_eval is not None:
                # fix random seeds for repetitive evaluation
                # this attribute should not be set during training
                torch.manual_seed(rep)

            for saint_data in self._graph_sampler:
                g = self.get_subgraph(saint_data)
                X, y = self.X[:, g['cent_n_id']], self.y[:, g['cent_n_id']]
                dataset_len = X.size(0)
                indices = list(range(dataset_len))

                if self.use_dist_sampler and dist.is_initialized():
                    # distributed sampler reference: torch.utils.data.distributed.DistributedSampler
                    if self.shuffle:
                        # ensure that all processes share the same permutated indices
                        tg = torch.Generator()
                        tg.manual_seed(self._epoch)
                        indices = torch.randperm(dataset_len, generator=tg).tolist()

                    world_size = dist.get_world_size()
                    node_rank = dist.get_rank()
                    num_samples_per_node = int(math.ceil(dataset_len * 1.0 / world_size))
                    total_size = world_size * num_samples_per_node

                    # add extra samples to make it evenly divisible
                    indices += indices[:(total_size - dataset_len)]
                    assert len(indices) == total_size

                    # get sub-batch for each process
                    # Node (rank=x) get [x, x+world_size, x+2*world_size, ...]
                    indices = indices[node_rank:total_size:world_size]
                    assert len(indices) == num_samples_per_node
                elif self.shuffle:
                    np.random.shuffle(indices)

                num_batches = (len(indices) + self.batch_size - 1) // self.batch_size
                for batch_id in range(num_batches):
                    start = batch_id * self.batch_size
                    end = (batch_id + 1) * self.batch_size
                    yield X[indices[start: end]], y[indices[start: end]], g, torch.LongTensor(indices[start: end])

    def get_length(self):
        if self.use_dist_sampler and dist.is_initialized():
            dataset_len = self.X.size(0)
            world_size = dist.get_world_size()
            num_samples_per_node = int(math.ceil(dataset_len * 1.0 / world_size))
        else:
            num_samples_per_node = self.X.size(0)
        length = (num_samples_per_node + self.batch_size - 1) // self.batch_size
        length *= len(self._graph_sampler)

        return length

    def __len__(self):
        return self._length

    def set_epoch(self, epoch):
        # self.set_epoch() will be called by BasePytorchTask on each epoch when using distributed training
        self._epoch = epoch
