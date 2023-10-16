import os
import random
from collections import namedtuple
import numpy as np
from scipy.stats import rankdata
from math import isnan, comb, floor, ceil
import networkx as nx
from copy import copy
import shutil

import torch
from torch_geometric.data import Data

from construct_leighton_graph import generate_leighton_graph
from construct_queen_graph import generate_queen_graph
from construct_mymethod_graph import generate_mymethod_graph
from construct_HC_dsatur_graph import generate_HC_dsatur_graph_as_networkx

MAX_RADIUS_FOR_NEIGHBOURS = 3

class Graph():
    def __init__(self, vertices_dict, edges_df):
        self.vertices = vertices_dict
        self.edges = edges_df


class MyGraphData(Data):    
    def __init__(self, orig_x=None, col_x=None, 
                orig_orig_edge_index=None, orig_orig_edge_attr=None,
                orig_col_edge_index=None, orig_col_edge_attr=None,
                col_orig_edge_index=None,  col_orig_edge_attr=None,
                **kwargs):
        
        super().__init__()
        self.orig_x = orig_x
        self.col_x = col_x

        self.orig_orig_edge_index = orig_orig_edge_index
        self.orig_orig_edge_attr = orig_orig_edge_attr

        self.orig_col_edge_index = orig_col_edge_index
        self.orig_col_edge_attr = orig_col_edge_attr
        
        self.col_orig_edge_index = col_orig_edge_index
        self.col_orig_edge_attr = col_orig_edge_attr

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'orig_orig_edge_index':
            return torch.tensor([[self.orig_x.shape[0]], [self.orig_x.shape[0]]])
        elif key == 'orig_col_edge_index':
            return torch.tensor([[self.orig_x.shape[0]], [self.col_x.shape[0]]])
        elif key == 'col_orig_edge_index':
            return torch.tensor([[self.col_x.shape[0]], [self.orig_x.shape[0]]])
        else:
            return super().__inc__(key, value, *args, **kwargs)
        

def save_training_stats_to_file(stats, filename):
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


def load_stats_from_file(filepath):
    with open(filepath) as file:
        lines = file.readlines()
        saved_episodes = lines[0].split(',')
        episode_rewards = lines[1].split(',')
        colours_used = lines[2].split(',')
        file.close()
    stats = namedtuple("Stats",["saved_episodes", "episode_rewards", "colours_used"])(
        saved_episodes = [float(item) for item in saved_episodes],
        episode_rewards=[float(item) for item in episode_rewards],
        colours_used=[float(item) for item in colours_used])
    return stats


def load_validation_stats_from_file(filepath):
    validation_stats = {}
    with open(filepath) as file:
        lines = file.readlines()
        if not len(lines)%3==0:
            raise Exception('No of lines in stats file should be multiple of 3')
        for ind in range(int(len(lines)/3)):
            stats = namedtuple("Stats",["episode_rewards", "colours_used"])(
                episode_rewards=[float(elt) for elt in lines[3*ind+1].split(',')],
                colours_used=[float(elt) for elt in lines[3*ind+2].split(',')])
            validation_stats[int(lines[3*ind])] = stats
        return validation_stats


def load_benchmark_stats_from_file(filepath):
    with open(filepath) as file:
        lines = file.readlines()
        episode_rewards = lines[0].split(',')
        colours_used = lines[1].split(',')
        file.close()
    stats = namedtuple("Stats",["episode_rewards", "colours_used"])(
        episode_rewards=[float(item) for item in episode_rewards],
        colours_used=[float(item) for item in colours_used])
    return stats


def sparse_graphs_match(edges1, edges2):
    # This function checks whether two sparse edge sets match
    # Only checks elts that aren't NaN in *either* edge set
    # So if one is NaN and the other isn't, function doesn't check that element
    found_mismatch = False
    for row_label in edges1.index.values:
        if found_mismatch:
            break
        for col_label in edges1.columns.values:
            elt1 = edges1.at[row_label, col_label]
            elt2 = edges2.at[row_label, col_label]
            if (not isnan(elt1)) and (not isnan(elt2)) and (elt1 != elt2):
                found_mismatch = True
                break
    return (not found_mismatch)

def sparse_graphs_match_inc_nans(edges1, edges2):
    # This function checks whether two sparse edge sets match
    # Only checks elts that aren't NaN in *both* edge sets
    # So if one is NaN and the other isn't, function returns False (i.e. the graphs dont' match)
    found_mismatch = False
    for row_label in edges1.index.values:
        if found_mismatch:
            break
        for col_label in edges1.columns.values:
            elt1 = edges1.at[row_label, col_label]
            elt2 = edges2.at[row_label, col_label]
            if not (isnan(elt1) and isnan(elt2)):
                if (elt1 != elt2):
                    found_mismatch = True
                    break
    return (not found_mismatch)

def sparse_graphs_match_exc_zeros(edges1, edges2):
    # This function checks whether two sparse edge sets match
    # Only checks elts which are not NaN or 0 in either edge sets 
    # This is primarily for the colouring task, in which the target edges are
    #   -1 if two edges should be assigned different colours
    #   0 otherwise

    for row_label in edges1.index.values:
        for col_label in edges1.columns.values:
            elt1 = edges1.at[row_label, col_label]
            elt2 = edges2.at[row_label, col_label]
            if (not isnan(elt1)) and (not isnan(elt2)) and (not elt1==0) and (not elt2==0) and (elt1 != elt2):
                return False
    return True

def colouring_graphs_match(target_edges, state_edges):
    # This function checks whether two edge sets representing colouring graphs match
    # Only checks elts which are -1 in the target edge set (i.e. those edges should be assigned different colours)
    # Recall in the colouring task, the target edges are
    #   NaN if the edge is a loop (i.e. a vertex to itself)
    #   -1 if the two vertices should be assigned different colours
    #   0 otherwise

    for row_label in target_edges.index.values:
        for col_label in target_edges.columns.values:
            target_elt = target_edges.at[row_label, col_label]
            # print('target_elt:')
            # print(target_elt)
            state_elt = state_edges.at[row_label, col_label]
            # print('state_elt:')
            # print(state_elt)
            
            if target_elt==-1:
                if not state_elt==-1:
                    return False
    return True


def graph_score(edge_attr, target_edge_attr):
    # using types of distance measure listed at http://www.iiisci.org/journal/pdv/sci/pdfs/GS315JG.pdf
    
    no_edges_in_target_edge_attr = 0
    no_corresponding_edges_in_edge_attr = 0
    
    for edge_attr_elt, target_edge_attr_elt in zip(edge_attr, target_edge_attr):
        if target_edge_attr_elt==-1:
            no_edges_in_target_edge_attr += 1
            if edge_attr_elt==-1:
                no_corresponding_edges_in_edge_attr+=1

    return no_corresponding_edges_in_edge_attr/no_edges_in_target_edge_attr


def all_objects_assigned(vertices):
    for vertex_value in vertices.values():
        if vertex_value == -1:
            return False
    return True


def current_edges_match_target(current_edge_attr, target_edge_attr):
    # Note this only checks that the edges that are -1 in the target are also -1 in the current graph
    # return all([current_edge_attr_value==-1 if target_edge_attr_value==-1 else 1 for current_edge_attr_value, target_edge_attr_value in zip(current_edge_attr, target_edge_attr)])
    for ind, target_edge_attr_value in enumerate(target_edge_attr):
        if target_edge_attr_value==-1 and current_edge_attr[ind]!=-1:
            return False
    return True


def current_edges_match_target_indices(current_edge_attr, target_edge_indices):
    # Note this only checks that the edges that are -1 in the target are also -1 in the current graph
    # This version only looks at the edges that are actually present in the target, and checks that they are -1 in current_edge_attr
    for ind in target_edge_indices:
        if current_edge_attr[ind]!=-1:
            return False
    return True


def edge_attr_match(edge_attr1, edge_attr2):
    return torch.equal(edge_attr1, edge_attr2)


def import_dimacs_graph(full_graph_filepath):
    
    f = open(full_graph_filepath, 'r')
    lines = f.read().splitlines()

    for i, line in enumerate(lines):
        x = line.split(' ')
        if x[0] == 'p':
            no_vertices = x[2]
            no_orig_vertices = int(no_vertices)
            orig_vertex_names = [i for i in range(no_orig_vertices)]
            target_edge_list = []
            target_edge_attr = []
            target_edge_indices = []
            ind=0
            edge_ind_dict = {vertex_name:{} for vertex_name in orig_vertex_names}
            for vertex_ind, vertex1_name in enumerate(orig_vertex_names):
                for vertex2_name in orig_vertex_names[vertex_ind+1:]:
                    if not vertex1_name==vertex2_name:
                        target_edge_attr.append(0)
                        target_edge_attr.append(0)
                        target_edge_list.append((vertex1_name, vertex2_name))
                        edge_ind_dict[vertex1_name][vertex2_name] = ind
                        ind+=1
                        target_edge_list.append((vertex2_name, vertex1_name))
                        edge_ind_dict[vertex2_name][vertex1_name] = ind
                        ind+=1
            
        elif x[0] == 'e':
            vertex1 = int(x[1])-1
            vertex2 = int(x[2])-1
            if not vertex1==vertex2:
                ind1 = edge_ind_dict[vertex1][vertex2]
                target_edge_attr[ind1] = -1
                target_edge_indices.append(ind1)
                ind2 = edge_ind_dict[vertex2][vertex1]
                target_edge_attr[ind2] = -1
                target_edge_indices.append(ind2)

    return no_orig_vertices, tuple(target_edge_list), tuple(target_edge_attr), edge_ind_dict, tuple(set(target_edge_indices))


def get_vertex_colour(neighbour_colours, possible_colour_names):
    # action_vertex_neighbours = [target['edge_list'][edge_ind][1] for edge_ind in target['target_edge_indices'] if target['edge_list'][edge_ind][0]==action_vertex]
    # neighbour_colours = set([current_vertex_values[vertex] for vertex in action_vertex_neighbours if not current_vertex_values[vertex]=='-1'])

    for colour_ind, colour in enumerate(possible_colour_names):
        if not colour in neighbour_colours:
            return colour_ind, colour

    raise Exception('Unable to find a colour to use.')


def convert_networkx_to_graph(vertices, edges):
    # converts a networkx graph from its default format to the format required for my method
    graph_nodes = list(vertices)
    graph_edges = list(edges)
    
    target_edge_list = []
    ind=0
    edge_ind_dict = {vertex_name:{} for vertex_name in graph_nodes}
    for vertex1_ind, vertex1_name in enumerate(graph_nodes):
        for vertex2_name in graph_nodes[vertex1_ind+1:]:
            target_edge_list.append((vertex1_name, vertex2_name))
            edge_ind_dict[vertex1_name][vertex2_name] = ind
            ind+=1
            target_edge_list.append((vertex2_name, vertex1_name))
            edge_ind_dict[vertex2_name][vertex1_name] = ind
            ind+=1

    target_edge_indices = []
    target_edge_attr = [0 for _ in target_edge_list]
    for vertex1, vertex2 in graph_edges:
        edge1_ind = edge_ind_dict[vertex1][vertex2]
        target_edge_attr[edge1_ind] = -1
        target_edge_indices.append(edge1_ind)
        edge2_ind = edge_ind_dict[vertex2][vertex1]
        target_edge_attr[edge2_ind] = -1
        target_edge_indices.append(edge2_ind)

    return tuple(target_edge_list), target_edge_attr, edge_ind_dict, target_edge_indices


def generate_barabasi_albert_graph(n):
    # need to specify m, the degree of each new vertex that is added
    # Choosing to make m between 2 and √n
    m = random.randint(2, int(n**0.5))
    ba_graph = nx.barabasi_albert_graph(n, m)
    return convert_networkx_to_graph(ba_graph.nodes, ba_graph.edges)


def generate_erdos_renyi_graph(n):
    # need to specify p, the probability of an edge being present
    # Choosing to make p between 0.1 and 0.9
    p = random.uniform(0.1, 0.9)
    er_graph = nx.erdos_renyi_graph(n, p)
    return convert_networkx_to_graph(er_graph.nodes, er_graph.edges)


def generate_watts_strogatz_graph(n):
    # need to specify:
    # k, the average degree
    # p, the probability of 'rewiring' each edge
    # Choosing to make k between 2 and √n
    # Choosing to make p between 0.1 and 0.3
    k = random.randint(2, int(n**0.5))
    p = random.uniform(0.1, 0.3)
    ws_graph = nx.watts_strogatz_graph(n, k, p)
    return convert_networkx_to_graph(ws_graph.nodes, ws_graph.edges)


def generate_gaussian_random_partition_graph(n):
    # need to specify:
    # s, the mean cluster size
    # v, a variable that deterines the variance of the cluster sizes
    # p_in, the probabilty of intra cluster connection
    # p_out, the probabilty of inter cluster connection
    # Choosing to make s between 2 and √n
    # Choosing to make v between √n and n/2
    # Choosing to make p_in between 0.5 and 1
    # Choosing to make p_out between 0 and 0.5
    s = random.randint(2, int(n**0.5))
    v = random.randint(int(n**0.5), int(n/2))
    p_in = random.uniform(0.5, 1)
    p_out = random.uniform(0, p_in/2)
    grp_graph = nx.gaussian_random_partition_graph(n, s, v, p_in, p_out)
    
    return convert_networkx_to_graph(grp_graph.nodes, grp_graph.edges)


def get_vertex_degrees(no_vertices, edge_list, target_edge_indices):
    vertex_degrees_aux = [0 for _ in range(no_vertices)]
    
    for elt in target_edge_indices:
        vertex1, vertex2 = edge_list[elt]
        vertex_degrees_aux[vertex1] += 1
        vertex_degrees_aux[vertex2] += 1
    vertex_degrees = tuple([int(elt/2) for elt in vertex_degrees_aux])
    return vertex_degrees


def get_vertex_colour_combos(min_no_colours_for_targets, max_no_colours_for_targets, min_no_vertices, max_no_vertices):
    leighton_vertex_colour_combos = []
    queen_vertex_colour_combos = []
    for no_colours in range(min_no_colours_for_targets, max_no_colours_for_targets+1):
        no_vertices = ceil(min_no_vertices/no_colours)*no_colours
        while no_vertices <= max_no_vertices:
            leighton_vertex_colour_combos.append((no_vertices, no_colours))
            if no_colours>=no_vertices**0.5 and no_colours <= no_vertices/3:
                queen_vertex_colour_combos.append((no_vertices, no_colours))
            no_vertices += no_colours
    # print('queen_vertex_colour_combos:')
    # for elt in queen_vertex_colour_combos:
    #     print(elt)
    # # print(queen_vertex_colour_combos)
    # input('press enter')

    return leighton_vertex_colour_combos, queen_vertex_colour_combos


def convert_graph_to_networkx(vertices, edges):
    graph = nx.Graph()
    
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)
    return graph


def generate_colour_groups_queen(board_dim, offset):
#     Note this function only works for square boards (with side length board_dim)
#     in which board_dim mod 6 = 1 or 5
#     offset is a number 0<offset<board_dim s.t. using it as the offset between rows will give a valid colouring
#     for board_dim=7, the possible offset values are 2,3,4,5
    col_group_to_uncoloured_vertices = {col_group:[] for col_group in range(board_dim)}
    vertex_to_col_group = {}
    for vertex in range(board_dim**2):
        col = vertex%board_dim
        row = int((vertex - col)/board_dim)
        
        col_group = (col-offset*row)%board_dim
        col_group_to_uncoloured_vertices[col_group].append(vertex)
        vertex_to_col_group[vertex] = col_group
        
    return col_group_to_uncoloured_vertices, vertex_to_col_group
        
        
def generate_action_list_queen(board_dim):
#     This function generates a list of actions for colouring a queen graph
#     Note this function only works for square boards (with side length board_dim)
#     in which board_dim mod 6 = 1 or 5
#     offset is a number 0<offset<board_dim s.t. using it as the offset between rows will give a valid colouring
#     for board_dim=7, the possible offset values are 2,3,4,5
    offset = 2
    col_group_to_uncoloured_vertices, vertex_to_col_group = generate_colour_groups_queen(board_dim, offset)
    
    action_list = []
    
    col_order = random.sample(list(col_group_to_uncoloured_vertices.keys()), len(col_group_to_uncoloured_vertices.keys()))
    for colour in col_order:
        action_list += random.sample(col_group_to_uncoloured_vertices[colour], len(col_group_to_uncoloured_vertices[colour]))
    
    return action_list


def import_dimacs_graph_as_networkx(full_graph_filepath):
    G = nx.Graph()
    
    f = open(full_graph_filepath, 'r')
    lines = f.read().splitlines()

    for i, line in enumerate(lines):
        x = line.split(' ')
        if x[0] == 'p':
            no_vertices = int(x[2])
            for vertex_ind in range(no_vertices):
                G.add_node(vertex_ind)
     
        elif x[0] == 'e':
            vertex1 = int(x[1])-1
            vertex2 = int(x[2])-1
            if not vertex1>=vertex2:
                G.add_edge(vertex1, vertex2)

    return G


def colour_graph_with_networkx(networkx_graph, strategy, repetitions=1):
    no_colours_by_ep = []
    if strategy == 'random_sequential':
        for _ in range(repetitions):
            d = nx.greedy_color(networkx_graph, strategy=strategy)
            no_colours = len(set(d.values()))
            no_colours_by_ep.append(no_colours)
    else:
        d = nx.greedy_color(networkx_graph, strategy=strategy)
        no_colours = len(set(d.values()))
        no_colours_by_ep.append(no_colours)
        
    no_colours_by_ep_array = np.array(no_colours_by_ep)
    min_no_colours = np.amin(no_colours_by_ep_array)
    ave_no_colours = np.mean(no_colours_by_ep_array)
    no_colours_std = np.std(no_colours_by_ep_array)

    return min_no_colours, ave_no_colours, no_colours_std


def colour_graph_with_all_networkx_strategies(networkx_graph, repetitions):
    # documentation for networkx colouring strategies can be found here: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.coloring.greedy_color.html#networkx.algorithms.coloring.greedy_color
    # It contains references that explain each of the strategies:
    
    # strategy='largest_first'
    # strategy='random_sequential'
    # strategy='smallest_last'
    # strategy='independent_set'
    # strategy='connected_sequential_bfs'
    # strategy='connected_sequential_dfs'
    # strategy='connected_sequential' #(alias for the previous strategy)
    # strategy='saturation_largest_first'
    # strategy='DSATUR' #(alias for the previous strategy)

    # note that det means the strategy is deterministic and stoch means it contains some stochasticity 
    # det strategies don't require repetitions but stoch strategies do
    strategies = ['random_sequential', 'largest_first', 'smallest_last', 'independent_set', 'connected_sequential_bfs', 'connected_sequential_dfs', 'saturation_largest_first']

    output = {}
    for strategy in strategies:
        min_no_colours, ave_no_colours, no_colours_std = colour_graph_with_networkx(networkx_graph, strategy, repetitions=1000)
        output[strategy] = {'min': min_no_colours, 'mean': ave_no_colours, 'std': no_colours_std}

    return output


def get_adj_matrix(no_vertices, present_edges):
    adj_matrix = np.zeros((no_vertices, no_vertices))
    for edge_start, edge_end in present_edges:
        adj_matrix[edge_start][edge_end] = 1
        
    return adj_matrix

def get_neighbour_distances(no_vertices, present_edges, adj_matrix):
    aux_neighbour_distances = np.ones((no_vertices, no_vertices)) * np.inf
    np.fill_diagonal(aux_neighbour_distances, 0)
    neighbour_distances = copy(aux_neighbour_distances)
    
    
    for _ in range(MAX_RADIUS_FOR_NEIGHBOURS):
        for edge_start, edge_end in present_edges:
            aux_neighbour_distances[edge_start] = np.minimum(np.minimum(neighbour_distances[edge_start], neighbour_distances[edge_end] + adj_matrix[edge_start][edge_end]), aux_neighbour_distances[edge_start])
        neighbour_distances = copy(aux_neighbour_distances)
        
    return neighbour_distances

def get_coloured_neighbour_distances(neighbour_distances, vertex_values):
    neighbour_distances_for_coloured_vertices = copy(neighbour_distances)
    for vertex, colour in vertex_values.items():
        if colour=='-1':
            neighbour_distances_for_coloured_vertices[:, vertex] = np.nan
    return neighbour_distances_for_coloured_vertices

def update_coloured_neighbour_distances(all_neighbour_distances, coloured_neighbour_distances, new_vertex):
    coloured_neighbour_distances[:, new_vertex] = all_neighbour_distances[:, new_vertex]
    return coloured_neighbour_distances

def get_uncoloured_neighbour_distances(neighbour_distances, vertex_values):
    neighbour_distances_for_uncoloured_vertices = copy(neighbour_distances)
    for vertex, colour in vertex_values.items():
        if colour!='-1':
            neighbour_distances_for_uncoloured_vertices[:, vertex] = np.nan
    return neighbour_distances_for_uncoloured_vertices


def update_uncoloured_neighbour_distances(uncoloured_neighbour_distances, new_vertex):
    uncoloured_neighbour_distances[:, new_vertex] = np.nan
    return uncoloured_neighbour_distances


def get_neighbour_distance_counts(no_vertices, min_path_lengths):
    
    count_neighbours_by_distance = np.zeros((no_vertices, MAX_RADIUS_FOR_NEIGHBOURS))
    for i in range(MAX_RADIUS_FOR_NEIGHBOURS):
        count_neighbours_by_distance[:, i] = np.count_nonzero(min_path_lengths == i+1, axis=1)

    return count_neighbours_by_distance


def all_vertices_assigned(vertex_values):
    # vertex_values is a dict mapping vertices to their values
    return not '-1' in vertex_values.values()


def all_vertices_assigned_using_data(data_x):
    # vertex_values is a dict mapping vertices to their values
    zero_tensor = torch.zeros(data_x.shape[1], dtype=torch.long)
    for elt in data_x:
        if torch.equal(elt, zero_tensor):
            return False
    return True


def all_colours_used(vertex_values, no_colours):
    return len(set(vertex_values.values())) == no_colours


def generate_graphs(graph_generation_dist, min_no_vertices, max_no_vertices, min_no_colours, max_no_colours, no_graphs=1):

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    if graph_generation_dist['leighton']>0 or graph_generation_dist['queen']>0:
        leighton_vertex_colour_combos, queen_vertex_colour_combos = get_vertex_colour_combos(min_no_colours, max_no_colours, min_no_vertices, max_no_vertices)

    generation_mechanism_list, generation_mechanism_probs = list(zip(*graph_generation_dist.items()))
    all_graphs = []
    
    no_graphs_generated = 0
    while no_graphs_generated < no_graphs:
        generation_mechanism = np.random.choice(generation_mechanism_list, p=generation_mechanism_probs)
        
        if generation_mechanism == 'mine':
            no_vertices = random.randrange(min_no_vertices, max_no_vertices+1)
            vertex_names = list(range(no_vertices))

            no_colours = random.randrange(min_no_colours, max_no_colours+1)
            colour_names = [str(i) for i in range(no_colours)]
            
            target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices = generate_mymethod_graph(vertex_names, colour_names)
        
        elif generation_mechanism == 'leighton':
            no_vertices, no_colours = random.choice(leighton_vertex_colour_combos)

            target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices = generate_leighton_graph(no_vertices, no_colours)

        elif generation_mechanism == 'queen':
            no_vertices, no_colours = random.choice(queen_vertex_colour_combos)

            target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices = generate_queen_graph(no_vertices, no_colours)

        elif generation_mechanism == 'BA':
            no_vertices = random.randrange(min_no_vertices, max_no_vertices+1)
            target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices = generate_barabasi_albert_graph(no_vertices)

        elif generation_mechanism == 'ER':
            no_vertices = random.randrange(min_no_vertices, max_no_vertices+1)
            target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices = generate_erdos_renyi_graph(no_vertices)

        elif generation_mechanism == 'WS':
            no_vertices = random.randrange(min_no_vertices, max_no_vertices+1)
            target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices = generate_watts_strogatz_graph(no_vertices)

        elif generation_mechanism == 'GRP':
            no_vertices = random.randrange(min_no_vertices, max_no_vertices+1)
            target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices = generate_gaussian_random_partition_graph(no_vertices)

        else:
            raise Exception('target_generation_mechanism should be in {mine, leighton, queen, BA, ER, WS, GRP}')

        graph_name = generation_mechanism + '_' + str(no_vertices)
        target = construct_target(no_vertices, target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices, name=graph_name)

        if target is None:
            continue
        else:
            no_graphs_generated += 1


        all_graphs.append(target)

    if len(all_graphs) == 1:
        return all_graphs[0]
    else:
        return all_graphs


def generate_HC_dsatur_graphs(min_n, max_n):
    all_graphs = []
    for n in range(min_n, max_n+1):
        nx_graph = generate_HC_dsatur_graph_as_networkx(n)
        no_vertices = nx_graph.number_of_nodes()
        target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices = convert_networkx_to_graph(nx_graph.nodes, nx_graph.edges)
        name='HC_dsatur_n_is_'+str(n)
        print('processing graph ', name)
        target = construct_target(no_vertices, target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices, name='HC_dsatur_n_is_'+str(n))
        all_graphs.append(target)

    return all_graphs


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


def construct_target(no_vertices, edge_list, edge_attr, edge_ind_dict, target_edge_indices, name=None):

    vertex_names = [i for i in range(no_vertices)]

    present_edges = tuple([edge_list[ind] for ind in target_edge_indices])
    
    networkx_graph = convert_graph_to_networkx(vertex_names, present_edges)
    if not nx.is_connected(networkx_graph):
        # print('graph ', name, ' is not connected')
        graph_connected = False
        # return None
    else:
        graph_connected = True

    def scale_dict(input_dict, scale_factor):
        scaled_dict = {key:val/scale_factor for key, val in input_dict.items()}
        return scaled_dict

    # vertex features 
    # (should be either list length n in same order as vertex names or dict keyed by vertex names)
    target_adj_matrix = get_adj_matrix(no_vertices, present_edges)
    neighbour_distances = get_neighbour_distances(no_vertices, present_edges, target_adj_matrix)
    neighbour_distance_counts = get_neighbour_distance_counts(no_vertices, neighbour_distances)
    neighbour_distance_counts_scaled = get_neighbour_distance_counts(no_vertices, neighbour_distances)/no_vertices
    clustering_coefficients = nx.clustering(networkx_graph)
    average_neighbor_degree = scale_dict(nx.average_neighbor_degree(networkx_graph), no_vertices)
    closeness_centrality = nx.closeness_centrality(networkx_graph)
    betweenness_centrality = nx.betweenness_centrality(networkx_graph)
    if graph_connected:
        current_flow_closeness_centrality = nx.current_flow_closeness_centrality(networkx_graph)
    else:
        current_flow_closeness_centrality = {i:1 for i in range(no_vertices)}
    if graph_connected:
        current_flow_betweenness_centrality = nx.current_flow_betweenness_centrality(networkx_graph)
    else:
        current_flow_betweenness_centrality = {i:1 for i in range(no_vertices)}
    eigenvector_centrality = nx.eigenvector_centrality(networkx_graph, max_iter=100000)
    # katz_centrality = nx.katz_centrality(networkx_graph, max_iter=100000)
    load_centrality = nx.load_centrality(networkx_graph)
    triangles = scale_dict(nx.triangles(networkx_graph), comb(no_vertices-1, 2))
    square_clustering = nx.square_clustering(networkx_graph)
    core_number = scale_dict(nx.core_number(networkx_graph), no_vertices)
    if graph_connected:
        eccentricity = scale_dict(nx.eccentricity(networkx_graph), no_vertices)
    else:
        eccentricity = {i:1 for i in range(no_vertices)}
    communicability_betweenness_centrality = nx.communicability_betweenness_centrality(networkx_graph)
    # the following are seemingly unbounded
    # (so potentially won't include their values in features but could still include their rank)
    closeness_vitality = nx.closeness_vitality(networkx_graph)
    subgraph_centrality = nx.subgraph_centrality(networkx_graph)

    # ranks
    def get_value_ranks(value_dict):
        list_by_name = [value_dict[name] for name in value_dict]
        return rankdata(list_by_name)

    average_neighbor_degree_rank = get_value_ranks(average_neighbor_degree)/no_vertices
    closeness_centrality_rank = get_value_ranks(closeness_centrality)/no_vertices
    betweenness_centrality_rank = get_value_ranks(betweenness_centrality)/no_vertices
    current_flow_closeness_centrality_rank = get_value_ranks(current_flow_closeness_centrality)/no_vertices
    current_flow_betweenness_centrality_rank = get_value_ranks(current_flow_betweenness_centrality)/no_vertices
    eigenvector_centrality_rank = get_value_ranks(eigenvector_centrality)/no_vertices
    # katz_centrality_rank = get_value_ranks(katz_centrality)/no_vertices
    load_centrality_rank = get_value_ranks(load_centrality)/no_vertices
    triangles_rank = get_value_ranks(triangles)/no_vertices
    square_clustering_rank = get_value_ranks(square_clustering)/no_vertices
    core_number_rank = get_value_ranks(core_number)/no_vertices
    eccentricity_rank = get_value_ranks(eccentricity)/no_vertices
    communicability_betweenness_centrality_rank = get_value_ranks(communicability_betweenness_centrality)/no_vertices
    closeness_vitality_rank = get_value_ranks(closeness_vitality)/no_vertices
    subgraph_centrality_rank = get_value_ranks(subgraph_centrality)/no_vertices

    # graph-level features
    no_edges = len(present_edges)/2
    edge_density = no_edges/(len(edge_list)/2)
    # degree_assortativity_coefficient = nx.degree_assortativity_coefficient(networkx_graph)
    transitivity = nx.transitivity(networkx_graph)
    if graph_connected:
        diameter = nx.diameter(networkx_graph)/no_vertices
        radius = nx.radius(networkx_graph)/floor((no_vertices-1)/2)
    else:
        diameter = 1
        radius = 0

    average_degrees = {
            'lin': np.mean(neighbour_distance_counts[:,0]),
            'log': np.mean(np.log(neighbour_distance_counts[:,0] + 1)),
            # 'exp': np.mean(np.exp(neighbour_distance_counts[:,0]))
            }

    # also include the averages of each of the measures above
    neighbour_distance_counts_scaled_avg = np.mean(neighbour_distance_counts_scaled, axis=0)
    clustering_coefficients_avg = np.mean(list(clustering_coefficients.values()))
    average_neighbor_degree_avg = np.mean(list(average_neighbor_degree.values()))
    closeness_centrality_avg = np.mean(list(closeness_centrality.values()))
    betweenness_centrality_avg = np.mean(list(betweenness_centrality.values()))
    current_flow_closeness_centrality_avg = np.mean(list(current_flow_closeness_centrality.values()))
    current_flow_betweenness_centrality_avg = np.mean(list(current_flow_betweenness_centrality.values()))
    eigenvector_centrality_avg = np.mean(list(eigenvector_centrality.values()))
    # katz_centrality_avg = np.mean(list(katz_centrality.values()))
    load_centrality_avg = np.mean(list(load_centrality.values()))
    triangles_avg = np.mean(list(triangles.values()))
    square_clustering_avg = np.mean(list(square_clustering.values()))
    core_number_avg = np.mean(list(core_number.values()))
    eccentricity_avg = np.mean(list(eccentricity.values()))
    communicability_betweenness_centrality_avg = np.mean(list(communicability_betweenness_centrality.values()))
    
    target = {
        'name': name,
        'no_vertices': no_vertices,
        'vertex_names': vertex_names,
        'edge_list': edge_list,
        'edge_attr': edge_attr,
        'edge_ind_dict': edge_ind_dict,
        'target_edge_indices': target_edge_indices,
        'present_edges': present_edges,
        'neighbour_distances': neighbour_distances,
        'neighbour_dist_counts': neighbour_distance_counts,
        'clustering_coeffs': clustering_coefficients,
        'average_neighbor_degree': average_neighbor_degree,
        'closeness_centrality': closeness_centrality,
        'betweenness_centrality': betweenness_centrality,
        'current_flow_closeness_centrality': current_flow_closeness_centrality,
        'current_flow_betweenness_centrality': current_flow_betweenness_centrality,
        'eigenvector_centrality': eigenvector_centrality,
        # 'katz_centrality': katz_centrality,
        'load_centrality': load_centrality,
        'triangles': triangles,
        'square_clustering': square_clustering,
        'core_number': core_number,
        'eccentricity': eccentricity,
        'communicability_betweenness_centrality': communicability_betweenness_centrality,
        'average_neighbor_degree_rank': average_neighbor_degree_rank,
        'closeness_centrality_rank': closeness_centrality_rank,
        'betweenness_centrality_rank': betweenness_centrality_rank,
        'current_flow_closeness_centrality_rank': current_flow_closeness_centrality_rank,
        'current_flow_betweenness_centrality_rank': current_flow_betweenness_centrality_rank,
        'eigenvector_centrality_rank': eigenvector_centrality_rank,
        # 'katz_centrality_rank': katz_centrality_rank,
        'load_centrality_rank': load_centrality_rank,
        'triangles_rank': triangles_rank,
        'square_clustering_rank': square_clustering_rank,
        'core_number_rank': core_number_rank,
        'eccentricity_rank': eccentricity_rank,
        'communicability_betweenness_centrality_rank': communicability_betweenness_centrality_rank,
        'closeness_vitality_rank': closeness_vitality_rank,
        'subgraph_centrality_rank': subgraph_centrality_rank,
        'no_edges': no_edges,
        'edge_density': edge_density,
        # 'degree_assortativity_coefficient': degree_assortativity_coefficient,
        'transitivity': transitivity,
        'diameter':  diameter,
        'radius': radius,
        'avg_deg': average_degrees,
        'neighbour_distance_counts_scaled_avg': neighbour_distance_counts_scaled_avg,
        'clustering_coefficients_avg': clustering_coefficients_avg,
        'average_neighbor_degree_avg': average_neighbor_degree_avg,
        'closeness_centrality_avg': closeness_centrality_avg,
        'betweenness_centrality_avg': betweenness_centrality_avg,
        'current_flow_closeness_centrality_avg': current_flow_closeness_centrality_avg,
        'current_flow_betweenness_centrality_avg': current_flow_betweenness_centrality_avg,
        'eigenvector_centrality_avg': eigenvector_centrality_avg,
        # 'katz_centrality_avg': katz_centrality_avg,
        'load_centrality_avg': load_centrality_avg,
        'triangles_avg': triangles_avg,
        'square_clustering_avg': square_clustering_avg,
        'core_number_avg': core_number_avg,
        'eccentricity_avg': eccentricity_avg,
        'communicability_betweenness_centrality_avg': communicability_betweenness_centrality_avg
    }

    return target


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

def save_all_files(base_dir):
    existing_main_filename = os.path.join('.', 'dqn_main_gn_colouring_v2_ultd.py')
    existing_agent_filename = os.path.join('.', 'dqn_agent_gn_colouring_v2_ultd.py')
    existing_env_filename = os.path.join('.', 'gym-graph-colouring-ultd-v2', 'gym_graph_colouring_ultd_v2', 'envs', 'graph_colouring_env_ultd_v2.py')
    existing_architecture_filename = os.path.join('.', 'architecture_colouring_v2_gn.py')
    existing_utils_filename = os.path.join('.', 'utils.py')
    copy_main_filename = os.path.join(base_dir, 'dqn_main_gn_colouring_v2_ultd.py')
    copy_agent_filename = os.path.join(base_dir, 'dqn_agent_gn_colouring_v2_ultd.py')
    copy_env_filename = os.path.join(base_dir, 'graph_colouring_env_ultd_v2.py')
    copy_architecture_filename = os.path.join(base_dir, 'architecture_colouring_v2_gn.py')
    copy_utils_filename = os.path.join(base_dir, 'utils.py')

    shutil.copyfile(existing_main_filename, copy_main_filename)
    shutil.copyfile(existing_agent_filename, copy_agent_filename)
    shutil.copyfile(existing_env_filename, copy_env_filename)
    shutil.copyfile(existing_architecture_filename, copy_architecture_filename)
    shutil.copyfile(existing_utils_filename, copy_utils_filename)

import warnings
warnings.filterwarnings("ignore")