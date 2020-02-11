import torch
try:
    from ._C import segment_graph as sg
except ImportError:
    from .pysrc.segment_graph import segment_graph as sg


def segment_graph(edge_index, edge_score, num_vertices, c, min_size=-1):
    return sg(edge_index, torch.argsort(edge_score).to(torch.int32), edge_score.to(torch.float32), num_vertices, c, min_size)
