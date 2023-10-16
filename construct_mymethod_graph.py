import random

def generate_mymethod_vertices(vertex_names, colour_names):
    # Takes the names of the vertices as input
    # and outputs vertex_values (a dict with colours for each vertex)
    #   Keys are vertex_names (of type int)
    #   Values are colour names (of type str)
    
    # Initially assigns one vertex to each colour (to ensure all colours have at least one vertex assigned to them)
    # Then assigns the remaining vertices to random colours

    unassigned_vertices = [vertex_name for vertex_name in vertex_names]
    vertex_values = {}

    # First ensure that all colours have at least one vertex assigned to them
    for colour_name in colour_names:
        random_vertex = random.choice(unassigned_vertices)
        vertex_values[random_vertex] = colour_name
        unassigned_vertices.remove(random_vertex)

    # Next randomly assign all remaining vertices
    for remaining_vertex in unassigned_vertices:
        colour_name = random.choice(colour_names)
        vertex_values[remaining_vertex] = colour_name

    return vertex_values


def generate_mymethod_graph(vertex_names, colour_names):
    # First uses generate_mymethod_vertices to generate a dict of the vertex values
    # Outputs edge_list, edge_attr, edge_ind_dict, target_edge_indices
    # edge_list is a list of 2-element lists (first and second elts represents start and end vertices resp) 
    #   As graphs are are modelled as complete, there are n(n-1) such elements
    # edge_attr is a list holding the attribute for the corresponding edge in edge_list
    # edge_ind_dict is a dict of dicts
    #   first key is start vertex
    #   second key is end vertex
    #   value gives the index of that edge in edge_list and edge_attr
    # target_edge_indices is a list holding the indices of edges that are present in the target graph
    #   i.e. the positions of -1 in edge_attr


    vertex_values = generate_mymethod_vertices(vertex_names, colour_names)

    # prob_of_including_edge = 1
    prob_of_including_edge = 2/3

    # At least one edge needs to be added to the target for it to be an acceptable target
    generated_ok_target = False
    while not generated_ok_target:

        edge_list = []
        edge_attr = []
        target_edge_indices = []

        ind=0
        
        # edge_ind_dict is a dict of dicts where edge_ind_dict[i][j] gives the position of the edge i-j in edge_list and edge_attr (of both current and target)
        edge_ind_dict = {vertex_name:{} for vertex_name in vertex_names}
        for vertex1_ind, vertex1_name in enumerate(vertex_names):
            for vertex2_name in vertex_names[vertex1_ind+1:]:
                if not vertex1_name==vertex2_name:
                    if (not ((vertex_values[vertex1_name] == '-1') or (vertex_values[vertex2_name] == '-1'))) and (vertex_values[vertex1_name] != vertex_values[vertex2_name]) and random.random()<prob_of_including_edge:
                        # if:
                        #   both vertices are assigned colours
                        #   and the vertices have different colours 
                        # then add vertex with probability prob_of_including_edge
                        edge_attr.append(-1)
                        edge_attr.append(-1)
                        generated_ok_target = True

                        edge_list.append((vertex1_name, vertex2_name))
                        edge_ind_dict[vertex1_name][vertex2_name] = ind
                        target_edge_indices.append(ind)
                        ind+=1
                        edge_list.append((vertex2_name, vertex1_name))
                        edge_ind_dict[vertex2_name][vertex1_name] = ind
                        target_edge_indices.append(ind)                            
                        ind+=1

                    else:
                        edge_attr.append(0)
                        edge_attr.append(0)

                        edge_list.append((vertex1_name, vertex2_name))
                        edge_ind_dict[vertex1_name][vertex2_name] = ind
                        ind+=1
                        edge_list.append((vertex2_name, vertex1_name))
                        edge_ind_dict[vertex2_name][vertex1_name] = ind
                        ind+=1

    return edge_list, edge_attr, edge_ind_dict, target_edge_indices


