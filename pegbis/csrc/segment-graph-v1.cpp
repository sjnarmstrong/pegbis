/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#ifndef SEGMENT_GRAPH
#define SEGMENT_GRAPH

#include <algorithm>
#include "disjoint-set.h"
#include <torch/extension.h>
#include <assert.h>


// threshold function
#define THRESHOLD(size, c) (c/size)

typedef struct {
  float w;
  long a, b;
} edge;

bool operator<(const edge &a, const edge &b) {
  return a.w < b.w;
}

/*
 * Segment a graph
 *
 * Returns a disjoint-set forest representing the segmentation.
 *
 * num_vertices: number of vertices in graph.
 * num_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for treshold function.
 */
torch::Tensor segment_graph(torch::Tensor edge_index, torch::Tensor edge_score, long num_vertices, float c, int min_size) {
  // check tensors are as expected
  assert(edge_index.dtype() == torch::kInt64);
  assert(edge_score.dtype() == torch::kFloat32);
  assert(edge_score.device().type() == torch::kCPU);
  assert(edge_index.device().type() == torch::kCPU);
  assert(edge_index.is_contiguous());
  assert(edge_score.is_contiguous());
  assert(edge_score.dim() == 1);
  assert(edge_index.dim() == 2);
  assert(edge_index.size(1) == edge_score.size(0));

  long num_edges = edge_index.size(1);
  // populate edges
  std::vector<edge> edges;
  float *edge_score_ptr = (float*)edge_score.data_ptr();
  long *edge_ind_ptr = (long*)edge_index.data_ptr();

  long j = 0;
  for (long i = 0; i < num_edges; i++) {
    edge pedge;
    pedge.w = edge_score_ptr[i];
    pedge.a = edge_ind_ptr[j++];
    pedge.b = edge_ind_ptr[j++];
    edges.push_back(pedge);;
    printf("Element at [%ld %ld]: %f\n", pedge.a, pedge.b, pedge.w);
  }

  // sort edges by weight
  std::sort(edges.begin(), edges.end());

  // make a disjoint-set forest
  universe u = universe(num_vertices);

  // init thresholds
  float *threshold = new float[num_vertices];
  std::fill_n(threshold, num_vertices, c);

  // for each edge, in non-decreasing weight order...
  //  for (int i = 0; i < num_edges; i++) {
  for(const edge edge_it : edges){
    // components conected by this edge
    long a = u.find(edge_it.a);
    long b = u.find(edge_it.b);
    if (a != b) {
      if ((edge_it.w <= threshold[a]) && (edge_it.w <= threshold[b])) {
        u.join(a, b);
        a = u.find(a);
        threshold[a] = edge_it.w + THRESHOLD(u.size(a), c);
      }
    }
  }

  if (min_size > 0){
    for(const edge edge_it : edges){
        // components conected by this edge
        long a = u.find(edge_it.a);
        long b = u.find(edge_it.b);
        if ((a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)) ) { u.join(a, b);}
    }
  }

  auto options = torch::TensorOptions()
    .dtype(torch::kInt64)
    .requires_grad(false);

  torch::Tensor res = torch::full({num_vertices}, -1, options);
  long *res_ptr = (long*)res.data_ptr();

  j = 0;
  for (long i = 0; i < num_edges; ++i) {
    long a = u.find(i);
    long clus_id = res_ptr[a];
    if (clus_id == -1) {
      clus_id = j++;
      res_ptr[a] = clus_id;
    }
    res_ptr[i] = clus_id;
  }


  // free up
  delete threshold;
  return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "C version of the efficient graph segmentation algorithm";
  m.def("segment_graph", &segment_graph, "Segment the graph");
}

#endif
