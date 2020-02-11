from pegbis.disjoint_set import *
import torch


# def segment_graph(num_vertices, num_edges, edges, c):
def segment_graph(edge_index, edge_score, num_vertices, c, min_size=-1):
    # sort edges by weight
    edge_argsort = torch.argsort(edge_score).tolist()
    # make a disjoint-set forest
    u = universe(num_vertices)
    # init thresholds
    threshold = torch.repeat_interleave(torch.tensor(c, dtype=torch.float), repeats=num_vertices)

    edge_index_cpu = edge_index.cpu()
    for edge_idx in edge_argsort:
        pedge = edge_index_cpu[:, edge_idx]

        # components connected by this edge
        a = u.find(pedge[0])
        b = u.find(pedge[1])
        ev = edge_score[edge_idx]
        if a != b:
            if (ev <= threshold[a]) and (ev <= threshold[b]):
                u.join(a, b)
                a = u.find(a)
                threshold[a] = ev + get_threshold(u.size(a), c)

    if min_size > 0:
        # post process small components
        for edge_idx in edge_argsort:
            pedge = edge_index_cpu[:, edge_idx]

            # components connected by this edge
            a = u.find(pedge[0])
            b = u.find(pedge[1])
            if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
                u.join(a, b)

    res = torch.full((num_vertices,), -1, dtype=torch.long)
    i = 0
    for j in range(num_vertices):
        a = u.find(j)
        clus_id = res[a]
        if clus_id == -1:
            clus_id = i
            i += 1
            res[a] = clus_id
        res[j] = clus_id

    return res


def get_threshold(size, c):
    return c / size

