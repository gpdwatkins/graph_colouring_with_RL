# import os
import pickle
# from datetime import datetime

# from utils import *
# from test_policy_functions import test_using_random_policy, test_using_dsatur, test_using_learned_policy
from test_policy_functions import test_using_learned_policy

class Dataset():
    # def __init__(self, graph_generation_distribution, min_no_vertices, max_no_vertices, min_no_colours, max_no_colours, max_radius_for_neighbours, no_graphs):
    def __init__(self, graphs):
        # init just creates an empty dataset
        # Use other functions to populate it with:
        # - randomly generated graphs, or
        # - a single graph (loaded from dimacs file) 
        # - a set of graphs (loaded from dimacs files)
        # Note - I want the class to be as lightweight as possible so where possible it's better to have the functions outside the class

        # self.graphs = self.populate_dataset(graph_generation_distribution, min_no_vertices, max_no_vertices, min_no_colours, max_no_colours, max_radius_for_neighbours, no_graphs)
        self.graphs = graphs

        # Get stats for random policy run on each graph in the dataset 
        # (note no_runs_per_graph determines the number of times to run it on each graph)
        # random_policy_stats_combined is a dict with keys:
        # - avg_max_episode_reward
        # - avg_episode_reward
        # - episode_reward_std
        # - avg_min_colours_used
        # - avg_colours_used
        # - colours_used_std
        # random_policy_stats_by_graph is a list of dicts, one for each graph, each with keys:
        # - name
        # - avg_episode_reward
        # - max_episode_reward
        # - episode_reward_std
        # - avg_colours_used
        # - colours_used_std
        # - min_colours_used 
        no_runs_per_graph = 100
        # no_runs_per_graph = 2

        self.avg_deg = get_avg_degrees(self.graphs)

        self.random_policy_stats_combined, self.random_policy_stats_by_graph, self.random_policy_stats_by_run = run_random_policy_on_dataset(self.graphs, no_runs_per_graph=no_runs_per_graph)
        
        # Get stats for dsatur run on each graph in the dataset 
        # dsatur_stats_combined is a dict with keys:
        # avg_episode_reward
        # - episode_reward_std
        # - avg_colours_used
        # - colours_used_std
        # dsatur_stats_by_graph is a list of dicts, one for each graph, each with keys:
        # - name
        # - episode_reward
        # - colours_used
        # - vertex_order
        self.dsatur_stats_combined, self.dsatur_stats_by_graph = run_dsatur_on_dataset(self.graphs)

#     # def populate_dataset(self, graph_generation_distribution, min_no_vertices, max_no_vertices, min_no_colours, max_no_colours, max_radius_for_neighbours, no_graphs):
#     #     # generate graphs according to some distribution
#     #     self.graphs = generate_graphs(graph_generation_distribution, min_no_vertices, max_no_vertices, min_no_colours, max_no_colours, max_radius_for_neighbours, no_graphs=no_graphs)


# def generate_random_dataset(graph_generation_distribution, min_no_vertices, max_no_vertices, min_no_colours, max_no_colours, no_graphs):
#     # constructs a dataset with graphs generated according to some distribution
#     graphs = generate_graphs(graph_generation_distribution, min_no_vertices, max_no_vertices, min_no_colours, max_no_colours, no_graphs=no_graphs)
#     return Dataset(graphs=graphs)


# def construct_dataset_from_dimacs_files(path):
#     if os.path.isdir(path):
#         graphs = []
#         for filename in os.listdir(path):
#             print('processing ', filename)
#             filepath = os.path.join(path, filename)
#             no_orig_vertices, edge_list, edge_attr, edge_ind_dict, target_edge_indices = import_dimacs_graph(filepath)
#             target = construct_target(no_orig_vertices, edge_list, edge_attr, edge_ind_dict, target_edge_indices, name=filename)
#             # if target is None:
#             #     print('filename:')
#             #     print(filename)
#             #     print('no_orig_vertices, edge_list, edge_attr, edge_ind_dict, target_edge_indices:')
#             #     print(no_orig_vertices, edge_list, edge_attr, edge_ind_dict, target_edge_indices)
#             graphs.append(target)
#         dataset = Dataset(graphs)
#     else:
#         no_orig_vertices, edge_list, edge_attr, edge_ind_dict, target_edge_indices = import_dimacs_graph(path)
#         target = construct_target(no_orig_vertices, edge_list, edge_attr, edge_ind_dict, target_edge_indices, name=os.path.split(path)[1])
#         dataset = Dataset([target])
#     return dataset


# def get_avg_degrees(graphs):
#     degs = []
#     for graph in graphs:
#         degs += get_vertex_degrees(graph['no_vertices'], graph['edge_list'], graph['target_edge_indices'])
#     degs = np.array(degs)
#     avg_degs: Dict[str, float] = {
#         'lin': np.mean(degs),
#         'log': np.mean(np.log(degs + 1)),
#         # 'exp': np.mean(np.exp(validation_degs))
#     }
#     return avg_degs


# def run_random_policy_on_dataset(graphs, no_runs_per_graph):
#     # returns a list of dicts (one for each graph) with keys:
#     # - avg_episode_reward
#     # - episode_reward_std
#     # - avg_colours_used
#     # - colours_used_std
#     stats_all_graphs, stats_by_graph = test_using_random_policy(graphs, runs_per_graph=no_runs_per_graph)
#     return stats_all_graphs, stats_by_graph
    

# def run_dsatur_on_dataset(graphs):
#     # returns a list of dicts (one for each graph in dataset) with keys:
#     # - avg_episode_reward
#     # - episode_reward_std
#     # - avg_colours_used
#     # - colours_used_std
#     stats_all_graphs, stats_by_graph = test_using_dsatur(graphs)
#     return stats_all_graphs, stats_by_graph


def run_learned_policy_on_dataset(graphs, agent, stochastic=False):
    # this function accepts an agent (with learned policy) and runs it on the graphs provided
    # use the stochastic argument to indicate whether to include randomness
    # if stochastic=True, use 100 runs on each graph
    
    # returns a list of dicts (one for each graph) with keys:
    # - avg_episode_reward
    # - episode_reward_std (will be 0 if stochastic==False)
    # - avg_colours_used
    # - colours_used_std (will be 0 if stochastic==False)
    stats_all_graphs, stats_by_graph = test_using_learned_policy(graphs, agent, stochastic=stochastic)
    return stats_all_graphs, stats_by_graph


# def save_dataset(dataset, filepath):
#     # save graphs (probably pickled) in some format
#     # dataset_dir = os.path.join(base_dir, dataset_type)
#     # os.mkdir(dataset_dir)
#     # dataset_filepath = os.path.join(dataset_dir, 'dataset.pickle')
#     pickle.dump(dataset, open(filepath, "wb" ))


def load_dataset(filepath):
    dataset = pickle.load( open(filepath, "rb"))
    return dataset


# if __name__ == "__main__":

#     # # ==================================================
#     # # GENERATE TRAINING/VALIDATION DATASETS OF GRAPHS FROM GIVEN DISTRIBUTION

#     # MIN_NO_VERTICES = 15
#     # MAX_NO_VERTICES = 50

#     # MIN_NO_COLOURS = 5
#     # MAX_NO_COLOURS = 15

#     # NO_TRAINING_GRAPHS = 1000
#     # NO_VALIDATION_GRAPHS = 100
#     # TRAINING_GRAPH_GENERATION_DISTRIBUTION = {'mine': 1/7, 'leighton': 1/7, 'queen': 1/7, 'BA': 1/7, 'ER': 1/7, 'WS': 1/7, 'GRP': 1/7}
#     # VALIDATION_GRAPH_GENERATION_DISTRIBUTION = TRAINING_GRAPH_GENERATION_DISTRIBUTION

#     # training_dataset = generate_random_dataset(TRAINING_GRAPH_GENERATION_DISTRIBUTION, MIN_NO_VERTICES, MAX_NO_VERTICES, MIN_NO_COLOURS, MAX_NO_COLOURS, NO_TRAINING_GRAPHS)
#     # validation_dataset = generate_random_dataset(VALIDATION_GRAPH_GENERATION_DISTRIBUTION, MIN_NO_VERTICES, MAX_NO_VERTICES, MIN_NO_COLOURS, MAX_NO_COLOURS, NO_VALIDATION_GRAPHS)
    
#     # start_time = datetime.now().strftime('%Y%m%d_%H%M')

#     # training_dataset_filepath = os.path.join('datasets', 'training_' + start_time + '.pickle')
#     # validation_dataset_filepath = os.path.join('datasets', 'validation_' + start_time + '.pickle')

#     # save_dataset(training_dataset, training_dataset_filepath)
#     # save_dataset(validation_dataset, validation_dataset_filepath)

#     # # ====================================
#     # # CONSTRUCT DATASET FROM DIRECTORY

#     # # full_graph_dir_path = os.path.join('colouring_graphs', 'lemos')
#     # # full_graph_dir_path = os.path.join('colouring_graphs', 'google_benchmark_graphs', 'h')
#     # full_graph_dir_path = os.path.join('colouring_graphs', 'queen_graphs')
#     # dimacs_graph_dataset = construct_dataset_from_dimacs_files(full_graph_dir_path)

#     # start_time = datetime.now().strftime('%Y%m%d_%H%M')

#     # dataset_filepath = os.path.join('datasets', 'queen_graphs_' + start_time + '.pickle')

#     # save_dataset(dimacs_graph_dataset, dataset_filepath)

#     # ====================================
#     # CONSTRUCT DATASET CONTAINING JUST test_graph.col

#     # full_graph_dir_path = os.path.join('colouring_graphs', 'test_graph')
#     # dimacs_graph_dataset = construct_dataset_from_dimacs_files(full_graph_dir_path)

#     # start_time = datetime.now().strftime('%Y%m%d_%H%M')

#     # dataset_filepath = os.path.join('datasets', 'test_graph_' + start_time + '.pickle')

#     # save_dataset(dimacs_graph_dataset, dataset_filepath)

#     # # ==================================================
#     # GENERATE DATASET OF GRAPHS FROM GIVEN DISTRIBUTION

#     # MIN_NO_VERTICES = 15
#     # MAX_NO_VERTICES = 50
#     # MIN_NO_VERTICES = 200
#     # MAX_NO_VERTICES = 200
#     MIN_NO_VERTICES = 15
#     MAX_NO_VERTICES = 64

#     MIN_NO_COLOURS = 5
#     MAX_NO_COLOURS = 15
#     # This version for constructing datasets of different sizes
#     MAX_NO_COLOURS = int(2*(MAX_NO_VERTICES**0.5))+1

#     NO_GRAPHS = 100
#     # GRAPH_GENERATION_DISTRIBUTION = {'mine': 1/7, 'leighton': 1/7, 'queen': 1/7, 'BA': 1/7, 'ER': 1/7, 'WS': 1/7, 'GRP': 1/7}
#     GRAPH_GENERATION_DISTRIBUTION = {'mine': 0, 'leighton': 0, 'queen': 1, 'BA': 0, 'ER': 0, 'WS': 0, 'GRP': 0}

#     start_time = datetime.now().strftime('%Y%m%d_%H%M')

#     dataset = generate_random_dataset(GRAPH_GENERATION_DISTRIBUTION, MIN_NO_VERTICES, MAX_NO_VERTICES, MIN_NO_COLOURS, MAX_NO_COLOURS, NO_GRAPHS)

#     dataset_filepath = os.path.join('datasets', 'just_queen_graphs' + start_time + '.pickle')
    
#     save_dataset(dataset, dataset_filepath)

#     # ====================================
#     # # CONSTRUCT DATASET CONTAINING HARD-TO-COLOUR GRAPHS FOR DSATUR

#     # MIN_n = 3
#     # MAX_n = 20

#     # start_time = datetime.now().strftime('%Y%m%d_%H%M')

#     # graphs = generate_HC_dsatur_graphs(MIN_n, MAX_n)
        
#     # dataset = Dataset(graphs)
#     # dataset_filepath = os.path.join('datasets', 'HC_dsatur_graphs' + start_time + '.pickle')
#     # save_dataset(dataset, dataset_filepath)
    
#     # ====================================

#     print('dataset generated')