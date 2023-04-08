import gym
from gym import spaces
from gym.utils import seeding

import torch
from torch_geometric.data import Data

from math import floor, isnan, ceil
import random
import numpy as np
from utils import *

import copy
from itertools import product

class GraphColouring(gym.Env):
    def __init__(self, dataset_graphs=None):
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Graph(node_space=spaces.Box(low=-np.inf, high=np.inf, shape=(1,)), edge_space=spaces.Box(low=-np.inf, high=np.inf, shape=(1,)))

        self.graphs = dataset_graphs


    def reset(self, target=None):
        # This function specifies a 'target graph' (the graph to be coloured)
        # and initialises current_graph with all vertices un-coloured
        # The target graph, self.target, is either:
        # - given as an input; or
        # - randomly selected from the dataset
        
        # self.target is a dict with keys ('edge_list', 'edge_attr', 'target_edge_indices')
        # edge_list is a list of 2-element lists (first and second elts represents start and end vertices resp) 
        #   As graphs are are modelled as complete, there are n(n-1) such elements
        # edge_attr is a list holding the attribute for the corresponding edge in edge_list
        # target_edge_indices is a list holding the indices of edges that are present in the target graph
        #   i.e. the positions of -1 in edge_attr
        
        # self.current_graph is a dict with keys ('vertex_values', 'edge_list', 'edge_attr')
        # vertex_values is a dict with vertex names (as int) as keys and colour names (as str) as values
        # edge_list is a list of 2-element lists (note that edge_list is identical in both current graph and target)
        # edge_attr is a list indexed using the edges 
        
        # Note that graphs are always modelled as complete (i.e. current_edge_index contains every edge-pair relationship)
        #   with edges that are present in the 'target' given attribute -1, and 0 otherwise

        self.episode_step = 0

        if target is None:
            target_ind = random.choice(range(len(self.graphs)))
            self.target = self.graphs[target_ind]
        else:
            self.target = target
        
        self.colour_order_for_ep = [i for i in range(self.target['no_vertices'])]

        self.current_graph = construct_initial_current_graph(self.target)

        if target is None:
            return target_ind, self.target, self.current_graph
        else:
            return self.target, self.current_graph


    def step(self, action_vertex):
        # Should take a vertex as input, colour it using the minimum available colour and output (current_graph, reward, done, info) 
        # This function updates self.current_graph
        
        self.episode_step += 1
        
        new_colour_ind, new_colour = get_vertex_colour(self.current_graph['neighbour_colours'][action_vertex], self.colour_order_for_ep)

        if not action_vertex in self.target['vertex_names']:
            raise Exception(action_vertex + ' is not a vertex name.')

        self.prev_no_colours_used = copy.copy(self.current_graph['no_colours_used'])
        self.prev_no_vertices_per_colour = copy.copy(self.current_graph['no_vertices_per_colour'])

        self.colour_vertex(action_vertex, new_colour_ind, new_colour)

        self.colour_leaf_nodes()
        
        self.done = (self.current_graph['no_uncoloured_vertices'] == 0)

        rewards = self.GenerateRewards(self.target['no_vertices'], self.current_graph['no_vertices_per_colour'], self.prev_no_vertices_per_colour, self.done, self.current_graph['no_colours_used'], prev_no_colours_used=self.prev_no_colours_used)

        episode_ended = self.done

        info = {'rewards': rewards, 'episode_ended': episode_ended}

        return (self.current_graph, sum(rewards.values()), self.done, info)


    def render(self, mode='human', close=False):
        print('target:')
        print(self.target_edges)
        print('current vertex values:')
        print(self.current_graph.vertices)
        print('current edge values:')
        print(self.current_graph.edges)


    def GenerateRewards(self, no_vertices, no_vertices_per_colour, prev_no_vertices_per_colour, done, no_colours_used, prev_no_colours_used=None):
        # Using whatever reward structure I decide on, return the reward 

        rewards = {}        

        rewards['new_colours_used'] = -1*(no_colours_used-prev_no_colours_used)

        return rewards


    def GenerateData_v1(self, target, current_graph):
        # Takes the target and the current_graph as input
        # Returns the graph as a torch_geometric Data object to use as input to the GN
        
        # vertex features contain:
        # - the vertex name (in this implementation vertex names are just 1,2,...,n)
        # - the current colour (note the colour names are fixed at the start of the episode; in this implementation they are 0,1,2,...)
        
        # edge features contain:
        #   -> -1 if vertices are connected by an edge in original graph
        #   -> 0 if not

        vertex_features = [
            [vertex_name] +
            [int(current_graph['vertex_values'][vertex_name])]
            for vertex_ind, vertex_name in enumerate(target['vertex_names'])
        ]

        global_features = [
        ]
        
        x = torch.tensor(vertex_features, dtype=torch.float, requires_grad=False)
        
        # For complete state representation:
        edge_index = torch.tensor(target['edge_list'], dtype=torch.long, requires_grad=False).transpose(0,1)
        edge_attr = torch.tensor([target['edge_attr']], dtype=torch.float, requires_grad=False).transpose(0,1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data, global_features


    def colour_vertex(self, action_vertex, new_colour_ind, new_colour):
        self.current_graph['vertex_values'][action_vertex] = new_colour

        if not new_colour in self.current_graph['no_vertices_per_colour'].keys():
            self.current_graph['no_vertices_per_colour'][new_colour] = 1
        else:
            self.current_graph['no_vertices_per_colour'][new_colour] += 1

        self.current_graph['assigned_vertices'].append(action_vertex)
        self.current_graph['assigned_vertices_onehot'][action_vertex] = 1
        self.current_graph['unassigned_vertices'] = [vertex for vertex in self.current_graph['unassigned_vertices'] if not vertex==action_vertex]
        self.current_graph['no_coloured_vertices'] += 1
        self.current_graph['no_uncoloured_vertices'] -= 1

        self.current_graph['max_colour_ind_used'] = max(int(new_colour_ind), self.current_graph['max_colour_ind_used'])
        self.current_graph['no_colours_used'] = self.current_graph['max_colour_ind_used'] + 1

        for other_vertex_name in self.target['vertex_names']:
            if not other_vertex_name == action_vertex:
                edge1_ind = self.target['edge_ind_dict'][action_vertex][other_vertex_name]
                edge2_ind = self.target['edge_ind_dict'][other_vertex_name][action_vertex]

                edge_feature = 2*(self.current_graph['vertex_values'][action_vertex] == self.current_graph['vertex_values'][other_vertex_name])-1 if not (self.current_graph['vertex_values'][other_vertex_name] == '-1') else 0

                self.current_graph['edge_attr'][edge1_ind] = edge_feature
                self.current_graph['edge_attr'][edge2_ind] = edge_feature

        self.current_graph['coloured_neighbour_distances'] = update_coloured_neighbour_distances(self.target['neighbour_distances'], self.current_graph['coloured_neighbour_distances'], action_vertex)
        self.current_graph['coloured_neighbour_distance_counts'] = get_neighbour_distance_counts(self.target['no_vertices'], self.current_graph['coloured_neighbour_distances'])

        uncoloured_neighbour_distances = update_uncoloured_neighbour_distances(self.current_graph['uncoloured_neighbour_distances'], action_vertex)
        self.current_graph['uncoloured_neighbour_distances'] = uncoloured_neighbour_distances
        uncoloured_neighbour_distance_counts = get_neighbour_distance_counts(self.target['no_vertices'], self.current_graph['uncoloured_neighbour_distances'])
        self.current_graph['uncoloured_neighbour_distance_counts'] = uncoloured_neighbour_distance_counts

        for neighbour_vertex in self.target['vertex_names']:
            if (action_vertex, neighbour_vertex) in self.target['present_edges']:
                self.current_graph['neighbour_colours'][neighbour_vertex].add(new_colour)
                self.current_graph['saturation'][neighbour_vertex] = len(self.current_graph['neighbour_colours'][neighbour_vertex])


    def colour_leaf_nodes(self):
        # Checks all the vertices in the graph
        # If the vertex has no uncoloured neighbours, assigns the minimum available colour to it
        for vertex in self.current_graph['unassigned_vertices']:
            if self.current_graph['uncoloured_neighbour_distance_counts'][vertex][0] == 0:
                new_colour_ind, new_colour = get_vertex_colour(self.current_graph['neighbour_colours'][vertex], self.colour_order_for_ep)
                self.colour_vertex(vertex, new_colour_ind, new_colour)
