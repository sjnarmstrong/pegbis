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
torch::Tensor segment_graph(torch::Tensor edge_index, torch::Tensor edge_order, torch::Tensor edge_score, int num_vertices, float c, int min_size) {
  int num_edges = edge_index.size(1);
  // populate edges
  auto edge_order_a = edge_order.accessor<int,1>();
  auto edge_ind_a = edge_index.accessor<long,2>();
  auto edge_score_a = edge_score.accessor<float,1>();

  // make a disjoint-set forest
  universe u = universe(num_vertices);

  // init thresholds
  float* threshold = new float[num_vertices];
  std::fill_n(threshold, num_vertices, c);

  // for each edge, in non-decreasing weight order...
  for (int i = 0; i < num_edges; i++)  {
    int p = edge_order_a[i];
    // components conected by this edge
    float w = edge_score_a[p];
    int a = u.find(edge_ind_a[0][p]);
    int b = u.find(edge_ind_a[1][p]);
    if (a != b) {
      if ((w <= threshold[a]) && (w <= threshold[b])) {
        u.join(a, b);
        a = u.find(a);
//        printf("(%f, %f, %f, %ld, %ld)", w, threshold[a], threshold[b], a, b);
        threshold[a] = w + THRESHOLD(u.size(a), c);
      }
    }
  }

  if (min_size > 0){
      for (int i = 0; i < num_edges; i++)  {
        int p = edge_order_a[i];
        // components conected by this edge
        int a = u.find(edge_ind_a[0][p]);
        int b = u.find(edge_ind_a[1][p]);
        if ((a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)) ) { u.join(a, b);}
    }
  }

  auto options = torch::TensorOptions()
    .dtype(torch::kInt64)
    .requires_grad(false);

  torch::Tensor res = torch::full({num_vertices}, -1, options);
  auto res_a = res.accessor<long,1>();

  long j = 0;
  for (int i = 0; i < num_vertices; i++) {
    int a = u.find(i);
    long clus_id = res_a[a];
    if (clus_id == -1) {
      clus_id = j++;
//      printf("[%ld:%ld->%ld]", i, a, j);
      res_a[a] = clus_id;
    }
    res_a[i] = clus_id;
  }


  // free up
  delete[] threshold;
  return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "C version of the efficient graph segmentation algorithm";
  m.def("segment_graph", &segment_graph, "Segment the graph");
}

#endif
