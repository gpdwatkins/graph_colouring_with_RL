import numpy as np
from copy import copy

MAX_RADIUS_FOR_NEIGHBOURS = 3


def save_stats_to_file(stats, filename):
    with open(filename, 'w') as file:
        file.write(','.join(np.array(stats.saved_episodes).astype(str)))
        file.write('\n')
        file.write(','.join(np.array(stats.episode_rewards).astype(str)))
        file.write('\n')
        file.write(','.join(np.array(stats.colours_used).astype(str)))
        file.close()


def save_validation_stats_to_file(validation_stats, filename):
    with open(filename, 'w') as file:
        for episode, stats in validation_stats.items():
            file.write(str(episode))
            file.write('\n')
            file.write(','.join(np.array(stats.episode_rewards).astype(str)))
            file.write('\n')
            file.write(','.join(np.array(stats.colours_used).astype(str)))
            file.write('\n')
        file.close()


def save_benchmark_stats_to_file(benchmark_stats, filename):
    with open(filename, 'w') as file:
        file.write(','.join(np.array(benchmark_stats.episode_rewards).astype(str)))
        file.write('\n')
        file.write(','.join(np.array(benchmark_stats.colours_used).astype(str)))
        file.close()


def get_vertex_colour(neighbour_colours, possible_colour_names):
    # action_vertex_neighbours = [target['edge_list'][edge_ind][1] for edge_ind in target['target_edge_indices'] if target['edge_list'][edge_ind][0]==action_vertex]
    # neighbour_colours = set([current_vertex_values[vertex] for vertex in action_vertex_neighbours if not current_vertex_values[vertex]=='-1'])

    for colour_ind, colour in enumerate(possible_colour_names):
        if not colour in neighbour_colours:
            return colour_ind, colour

    raise Exception('Unable to find a colour to use.')


def get_vertex_degrees(no_vertices, edge_list, target_edge_indices):
    vertex_degrees_aux = [0 for _ in range(no_vertices)]
    
    for elt in target_edge_indices:
        vertex1, vertex2 = edge_list[elt]
        vertex_degrees_aux[vertex1] += 1
        vertex_degrees_aux[vertex2] += 1
    vertex_degrees = tuple([int(elt/2) for elt in vertex_degrees_aux])
    return vertex_degrees


def update_coloured_neighbour_distances(all_neighbour_distances, coloured_neighbour_distances, new_vertex):
    coloured_neighbour_distances[:, new_vertex] = all_neighbour_distances[:, new_vertex]
    return coloured_neighbour_distances


def update_uncoloured_neighbour_distances(uncoloured_neighbour_distances, new_vertex):
    uncoloured_neighbour_distances[:, new_vertex] = np.nan
    return uncoloured_neighbour_distances


def get_neighbour_distance_counts(no_vertices, min_path_lengths):
    
    count_neighbours_by_distance = np.zeros((no_vertices, MAX_RADIUS_FOR_NEIGHBOURS))
    for i in range(MAX_RADIUS_FOR_NEIGHBOURS):
        count_neighbours_by_distance[:, i] = np.count_nonzero(min_path_lengths == i+1, axis=1)

    return count_neighbours_by_distance


def generate_edge_features_from_vertex_features(edge_list, vertex_values):
        # Takes a list of edges (note this is complete) and the vertex values as input
        # and outputs the edge_attr for that graph

        # Note that only edges that are specified in the target are included
        # i.e. target_edge_index is used as-is for the edge_index

        # The vertex-vertex edges values are: 
        # +1 if their vertex values match
        # -1 if they don't match
        # 0 if either of the vertices is unassigned (i.e. either vertex value is -1)

        edge_attr = [2*(vertex_values[vertex1] == vertex_values[vertex2])-1 if not ((vertex_values[vertex1] == '-1') or (vertex_values[vertex2] == '-1')) else 0 for (vertex1, vertex2) in edge_list]
        return edge_attr
        

def construct_initial_current_graph(target_graph):
    
    vertex_values = {vertex:'-1' for vertex in target_graph['vertex_names']}
    
    current_edge_attr = generate_edge_features_from_vertex_features(target_graph['edge_list'], vertex_values)

    neighbour_colours = {v: set() for v in target_graph['vertex_names']}
    # note no_distinct_neighbour_colours is also called saturation, in the DSATUR algo
    no_distinct_neighbour_colours = {v: 0 for v in target_graph['vertex_names']}

    coloured_neighbour_distances = np.ones((target_graph['no_vertices'], target_graph['no_vertices'])) * np.inf
    coloured_neighbour_distance_counts = np.zeros((target_graph['no_vertices'], MAX_RADIUS_FOR_NEIGHBOURS))
    
    uncoloured_neighbour_distances = copy(target_graph['neighbour_distances'])
    uncoloured_neighbour_distance_counts = copy(target_graph['neighbour_dist_counts'])

    assigned_vertices = []
    assigned_vertices_onehot = [0 for vertex in target_graph['vertex_names']]
    unassigned_vertices = [vertex for vertex in target_graph['vertex_names']]

    current_graph = {
            'vertex_values': vertex_values,
            'edge_attr': current_edge_attr,
            'neighbour_colours': neighbour_colours,
            'coloured_neighbour_distances': coloured_neighbour_distances,
            'coloured_neighbour_distance_counts': coloured_neighbour_distance_counts,
            'uncoloured_neighbour_distances': uncoloured_neighbour_distances,
            'uncoloured_neighbour_distance_counts': uncoloured_neighbour_distance_counts,
            'saturation': no_distinct_neighbour_colours,
            'assigned_vertices': assigned_vertices,
            'assigned_vertices_onehot': assigned_vertices_onehot,
            'unassigned_vertices': unassigned_vertices,
            'no_coloured_vertices': len(assigned_vertices),
            'no_uncoloured_vertices': len(unassigned_vertices),
            'no_vertices_per_colour': {},
            'no_colours_used': 0,
            'max_colour_ind_used': -1
        }
    
    return current_graph

import warnings
warnings.filterwarnings("ignore")