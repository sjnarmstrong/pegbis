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
torch::Tensor segment_graph(torch::Tensor edge_index, torch::Tensor edge_order, torch::Tensor edge_score, long num_vertices, float c, int min_size) {
  // check tensors are as expected
  assert(edge_index.dtype() == torch::kInt64);
  assert(edge_order.dtype() == torch::kInt32);
  assert(edge_score.dtype() == torch::kFloat64);
  assert(edge_order.device().type() == torch::kCPU);
  assert(edge_index.device().type() == torch::kCPU);
  assert(edge_score.device().type() == torch::kCPU);
  assert(edge_index.is_contiguous());
  assert(edge_order.is_contiguous());
  assert(edge_score.is_contiguous());
  assert(edge_order.dim() == 1);
  assert(edge_index.dim() == 2);
  assert(edge_score.dim() == 1);
  assert(edge_index.size(1) == edge_order.size(0));
  assert(edge_index.size(1) == edge_score.size(0));

  long num_edges = edge_index.size(1);
  // populate edges
  long *edge_order_ptr = (long*)edge_order.data_ptr();
  long *edge_ind_ptr = (long*)edge_index.data_ptr();
  double *edge_score_ptr = (double*)edge_score.data_ptr();

  // make a disjoint-set forest
  universe u = universe(num_vertices);

  // init thresholds
  float threshold[num_vertices];
  std::fill_n(threshold, num_vertices, c);

  // for each edge, in non-decreasing weight order...
  for (long i = 0; i < num_edges; i++)  {
    // components conected by this edge
    long p = edge_order_ptr[i];
    long v1 = edge_ind_ptr[p];
    long v2 = edge_ind_ptr[p+num_edges];
    float w = edge_score_ptr[p];
    long a = u.find(v1);
    long b = u.find(v2);
    if (a != b) {
      if ((w <= threshold[a]) && (w <= threshold[b])) {
        u.join(a, b);
        a = u.find(a);
        threshold[a] = w + THRESHOLD(u.size(a), c);
      }
    }
  }

  if (min_size > 0){
    for (long i = 0; i < num_edges; i++)  {
        // components conected by this edge
        long p = edge_order_ptr[i];
        long v1 = edge_ind_ptr[p];
        long v2 = edge_ind_ptr[p+num_edges];
        long a = u.find(v1);
        long b = u.find(v2);
        if ((a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)) ) { u.join(a, b);}
    }
  }

  auto options = torch::TensorOptions()
    .dtype(torch::kInt64)
    .requires_grad(false);

  torch::Tensor res = torch::full({num_vertices}, -1, options);
  long *res_ptr = (long*)res.data_ptr();

  long j = 0;
  for (long i = 0; i < num_vertices; i++) {
    long a = u.find(i);
    long clus_id = res_ptr[a];
    if (clus_id == -1) {
      clus_id = j++;
      res_ptr[a] = clus_id;
    }
    res_ptr[i] = clus_id;
  }


  // free up
  return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "C version of the efficient graph segmentation algorithm";
  m.def("segment_graph", &segment_graph, "Segment the graph");
}

#endif
