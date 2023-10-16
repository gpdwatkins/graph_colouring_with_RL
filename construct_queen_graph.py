import random

def generate_queen_graph(n, k):
# generate a queens graph

    if not n%k==0:
        raise Exception('n must be a multiple of k for a Queen graphs')

    if random.random() > 0.5:
        no_rows = k
        no_cols = n/k
    else:
        no_cols = k
        no_rows = n/k

    orig_vertex_names = [i for i in range(n)]
    target_edge_list = []
    target_edge_attr = []
    target_edge_indices = []
    ind=0
    edge_ind_dict = {vertex_name:{} for vertex_name in orig_vertex_names}

    for vertex1_ind, vertex1_name in enumerate(orig_vertex_names):
        for vertex2_name in orig_vertex_names[vertex1_ind+1:]:
            target_edge_list.append((vertex1_name, vertex2_name))
            edge_ind_dict[vertex1_name][vertex2_name] = ind
            ind+=1
            target_edge_list.append((vertex2_name, vertex1_name))
            edge_ind_dict[vertex2_name][vertex1_name] = ind
            ind+=1
            
            if not vertex1_name==vertex2_name:
                vertex1_row = int(vertex1_name/no_cols)
                vertex2_row = int(vertex2_name/no_cols)
                if vertex1_row == vertex2_row:
                    # same row
                    target_edge_attr.append(-1)
                    ind1 = edge_ind_dict[vertex1_name][vertex2_name]
                    target_edge_indices.append(ind1)
                    target_edge_attr.append(-1)
                    ind2 = edge_ind_dict[vertex2_name][vertex1_name]
                    target_edge_indices.append(ind2)
                else:
                    vertex1_col = vertex1_name%no_cols
                    vertex2_col = vertex2_name%no_cols
                    if vertex1_col == vertex2_col:
                        # same column
                        target_edge_attr.append(-1)
                        ind1 = edge_ind_dict[vertex1_name][vertex2_name]
                        target_edge_indices.append(ind1)
                        target_edge_attr.append(-1)
                        ind2 = edge_ind_dict[vertex2_name][vertex1_name]
                        target_edge_indices.append(ind2)
                    else:
                        if abs(vertex2_row-vertex1_row)==abs(vertex2_col-vertex1_col):
                            # same diagonal
                            target_edge_attr.append(-1)
                            ind1 = edge_ind_dict[vertex1_name][vertex2_name]
                            target_edge_indices.append(ind1)
                            target_edge_attr.append(-1)
                            ind2 = edge_ind_dict[vertex2_name][vertex1_name]
                            target_edge_indices.append(ind2)
                        else:
                            target_edge_attr.append(0)
                            target_edge_attr.append(0)

    return target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices