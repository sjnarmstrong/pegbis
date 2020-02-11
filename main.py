from PIL import Image
import numpy as np
import torch
from pegbis.segment import segment_graph
import time
import random


# randomly creates RGB
def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(0, 255)
    rgb[1] = random.randint(0, 255)
    rgb[2] = random.randint(0, 255)
    return rgb


def flatten_data(image):
    width, height = image.size
    image_torch = torch.from_numpy(np.array(image).astype(np.float))
    feats = torch.empty(width * height, 3, dtype=torch.float)
    pos = torch.empty(width * height, 2, dtype=torch.float)
    edges_size = width * height * 5
    edges = torch.empty(2, edges_size, dtype=torch.long)
    num = 0
    for y in range(height):
        for x in range(width):
            feats[y * width + x] = image_torch[y, x]
            pos[y * width + x] = torch.tensor([y, x])
            if x < width - 1:
                edges[0, num] = int(y * width + x)
                edges[1, num] = int(y * width + (x + 1))
                num += 1
            if y < height - 1:
                edges[0, num] = int(y * width + x)
                edges[1, num] = int((y + 1) * width + x)
                num += 1

            if (x < width - 1) and (y < height - 2):
                edges[0, num] = int(y * width + x)
                edges[1, num] = int((y + 1) * width + (x + 1))
                num += 1

            if (x < width - 1) and (y > 0):
                edges[0, num] = int(y * width + x)
                edges[1, num] = int((y - 1) * width + (x + 1))
                num += 1

    return feats, pos, edges[:, :num]


def get_edges(ft_pos, width, height, x_offset=0, y_offset=0):
    if x_offset > 0:
        ft_pos = ft_pos[ft_pos[:, 1] < width-x_offset]
    if x_offset < 0:
        ft_pos = ft_pos[ft_pos[:, 1] >= -x_offset]
    if y_offset > 0:
        ft_pos = ft_pos[ft_pos[:, 0] < height-y_offset]
    if y_offset < 0:
        ft_pos = ft_pos[ft_pos[:, 0] >= -y_offset]
    i_1 = torch.tensor([[width, 1]]) @ ft_pos.T
    i_2 = torch.tensor([[width, 1]]) @ (ft_pos.T+torch.tensor([[y_offset], [x_offset]]))
    return torch.stack((i_1[0], i_2[0]))


def load_image(path):
    image = Image.open(path)
    x = torch.from_numpy(np.array(image).astype(np.float))
    pos = torch.from_numpy(np.indices(x.shape[:-1])).flatten(1).T
    x = x.flatten(0,1)
    width, height = image.size
    edge_arr = []
    edge_arr += [get_edges(pos, width, height, 0, 0)]

    edge_arr += [get_edges(pos, width, height, -1, 0)]
    edge_arr += [get_edges(pos, width, height, 0, -1)]
    edge_arr += [get_edges(pos, width, height, -1, -1)]
    edge_arr += [get_edges(pos, width, height, -1, 1)]

    edge_arr += [get_edges(pos, width, height, 1, 0)]
    edge_arr += [get_edges(pos, width, height, 0, 1)]
    edge_arr += [get_edges(pos, width, height, 1, 1)]
    edge_arr += [get_edges(pos, width, height, 1, -1)]
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return x, torch.cat(edge_arr, dim=1), batch, pos, image


x, edges, batch, pos, image = load_image('data/BigTree.jpg')
err = torch.norm(x[edges[0]]-x[edges[1]], dim=1)

w,h = image.size

# t0 = time.time()
# cluster = segment_graph_c(edges, torch.argsort(err), err, w*h, 500, 50)
# print(f"elapsed time: {time.time()- t0}")
for i in range(30):
    t0 = time.time()
    cluster = segment_graph(edges, err, w*h, 500, 50)
    print(f"elapsed time2: {time.time()- t0}")

output = np.zeros(shape=(h, w, 3))
output2 = np.zeros(shape=(h, w, 3))

# pick random colors for each component
cmax = cluster.max()+1
colors = np.zeros(shape=(cmax, 3))
for i in range(cmax):
    colors[i, :] = random_rgb()

for i, p in enumerate(pos.to(dtype=torch.int)):
    output[p[0], p[1], :] = colors[cluster[i], :]
    output2[p[0], p[1], :] = x[i]

Image.fromarray(output.astype(np.uint8)).show()
Image.fromarray(output2.astype(np.uint8)).show()