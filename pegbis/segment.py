import torch
try:
    from ._C import segment_graph as sg
except ImportError:
    from .pysrc.segment_graph import segment_graph as sg


def segment_graph(edge_index, edge_score, num_vertices, c, min_size=-1):
    return sg(edge_index, torch.argsort(edge_score), edge_score, num_vertices, c, min_size)
