# # import gym_graph_colouring_ultd_v1
import gym
# import os
import numpy as np
# from collections import namedtuple
from utils import MAX_RADIUS_FOR_NEIGHBOURS
# import gym_graph_colouring_ultd_v2

# def choose_random_action(unassigned_vertices):
#     vertex_ind = random.choice(unassigned_vertices)
#     return vertex_ind

# def test_using_random_policy(graphs, runs_per_graph=1):

#     test_env = gym.make('graph-colouring-ultd-v2')

#     stats_by_graph = []
#     episode_rewards_all_graphs = []
#     colours_used_all_graphs = []
#     max_episode_rewards_by_graph = []
#     min_colours_used_by_graph = []

#     colours_used_by_run = {i:[] for i in range(runs_per_graph)}

#     for graph_target in graphs:
#         # print('running random policy on ', graph_target['name'])
#         print('processing ', graph_target['name'])
#         episode_rewards=[]
#         colours_used = []
#         max_episode_reward = -np.inf
#         min_colours_used = np.inf

#         for repeat in range(runs_per_graph):
            
#             ep_ended = False
#             ep_score = 0

#             target, current_graph = test_env.reset(target=graph_target)
            
#             while not ep_ended:

#                 vertex_index = choose_random_action(current_graph['unassigned_vertices'])
                
#                 next_graph, reward, done, info = test_env.step(vertex_index)
                
#                 current_graph = next_graph
                
#                 if 'episode_ended' in info.keys():
#                     ep_ended = info['episode_ended']
#                 ep_ended = ep_ended or done

#                 ep_score += reward
            
#             episode_rewards.append(ep_score)
#             colours_used.append(current_graph['no_colours_used'])
#             max_episode_reward = max(max_episode_reward, ep_score)
#             min_colours_used = min(min_colours_used, current_graph['no_colours_used'])

#             colours_used_by_run[repeat].append(current_graph['no_colours_used'])

#         # stats_for_graph = namedtuple("Stats",["episode_rewards", "avg_episode_reward", "episode_reward_std", "colours_used", "avg_colours_used", "colours_used_std"])(
#         #     episode_rewards=episode_rewards,
#         #     avg_episode_reward=np.mean(episode_rewards),
#         #     episode_reward_std=np.std(episode_rewards),
#         #     colours_used = colours_used,
#         #     avg_colours_used=np.mean(colours_used),
#         #     colours_used_std=np.std(colours_used)
#         # )

#         episode_rewards_all_graphs += episode_rewards
#         colours_used_all_graphs += colours_used
#         max_episode_rewards_by_graph.append(max_episode_reward)
#         min_colours_used_by_graph.append(min_colours_used)

#         stats_for_graph = {
#             "name": graph_target['name'],
#             "avg_episode_reward": np.mean(episode_rewards),
#             "max_episode_reward": max_episode_reward,
#             "episode_reward_std": np.std(episode_rewards),
#             "avg_colours_used": np.mean(colours_used),
#             "colours_used_std": np.std(colours_used),
#             "min_colours_used": min_colours_used
#         }

#         stats_by_graph.append(stats_for_graph)

#     stats_by_run = []
    
#     for key, value in colours_used_by_run.items():
#         stats_for_run = {
#             "run_no": key,
#             "avg_colours_used": np.mean(value),
#             "colours_used_std": np.std(value)
#         }
#         stats_by_run.append(stats_for_run)

#     stats_all_graphs = {
#         "avg_max_episode_reward": np.mean(max_episode_rewards_by_graph),
#         "avg_episode_reward": np.mean(episode_rewards_all_graphs),
#         "episode_reward_std": np.std(episode_rewards_all_graphs),
#         "avg_min_colours_used": np.mean(min_colours_used_by_graph),
#         "avg_colours_used": np.mean(colours_used_all_graphs),
#         "colours_used_std": np.std(colours_used_all_graphs)
#     }

#     return stats_all_graphs, stats_by_graph, stats_by_run

# def test_using_dsatur(graphs):
    
#     env = gym.make('graph-colouring-ultd-v2', dataset_graphs=graphs)
    
#     from networkx.algorithms.coloring.greedy_coloring import strategy_saturation_largest_first

#     stats_by_graph = []
#     episode_rewards_all_graphs = []
#     colours_used_all_graphs = []
    
#     for graph_target in graphs:
#         ep_score = 0
#         # if get_trajectory:
#         #     trajectory = []
        
#         networkx_graph = convert_graph_to_networkx(graph_target['vertex_names'], graph_target['present_edges'])
        
#         target, current_graph = env.reset(target=graph_target)
        
#         # if get_trajectory:
#         #     current_data, current_global_features = env.GenerateData_v1(target, current_graph)

#         vertex_order = []

#         multiplier = np.array([[target['no_vertices']**(MAX_RADIUS_FOR_NEIGHBOURS-1-i) for i in range(MAX_RADIUS_FOR_NEIGHBOURS)] for _ in range(target['no_vertices'])])
#         first_vertex = np.argmax(np.sum(target['neighbour_dist_counts'] * multiplier, axis=1))

#         vertex_order.append(first_vertex)
    
#         next_graph, reward, done, info = env.step(first_vertex)
#         ep_score += reward
#         next_data, next_global_features = env.GenerateData_v1(target, next_graph)

#         # if get_trajectory:
#         #     trajectory.append(tuple([current_data, current_global_features, np.array(next_graph['assigned_vertices_onehot']), first_vertex, reward+1, next_data, next_global_features, int(done)]))

#         # print('new_assignment: ', env.current_graph['vertex_values'])
#         # print('reward: ', reward)
#         # print('done: ', done)
#         # print('\n')

#         colors = {vertex:colour for vertex, colour in next_graph['vertex_values'].items() if not colour == '-1'}
#         node_generator = strategy_saturation_largest_first(networkx_graph, colors)
        
#         current_graph = next_graph
#         # if get_trajectory:
#         #     current_data = next_data
#         #     current_global_features = next_global_features

#         for vertex in node_generator:
#             if current_graph['vertex_values'][vertex] == '-1':
#                 vertex_order.append(vertex)
                
#                 next_graph, reward, done, info = env.step(vertex)
#                 ep_score += reward
#                 # if get_trajectory:
#                 #     next_data, next_global_features = env.GenerateData_v1(target, next_graph)
#                 # Set to keep track of colors of neighbours
#                 neighbour_colors = {colors[v] for v in networkx_graph[vertex] if v in colors}
#                 # # Find the first unused color.
#                 # for color in itertools.count():
#                 #     if color not in neighbour_colors:
#                 #         break
#                 # agent.expert_remember(current_data, current_global_features, np.array(next_graph['assigned_vertices_onehot']), vertex, reward+1, next_data, next_global_features, int(done))

#                 # print('new_assignment: ', env.current_graph['vertex_values'])
#                 # print('reward: ', reward)
#                 # print('done: ', done)
#                 # print('\n')
            
#             else:
#                 raise Exception('should not have got here')

#             # if get_trajectory:
#             #     trajectory.append(tuple([current_data, current_global_features, np.array(next_graph['assigned_vertices_onehot']), vertex, reward+1, next_data, next_global_features, int(done)]))

#             # Get current colour assignments
#             for vertex, colour in next_graph['vertex_values'].items():
#                 if not colour == '-1':
#                     colors[vertex] = colour

#             # print('\ncurrent_data.x:')
#             # print(current_data.x)
#             # print('\ncurrent_global_features:')
#             # print(current_global_features)
#             # print('\nassigned_vertices_onehot:')
#             # print(np.array(next_graph['assigned_vertices_onehot']))
#             # print('\nvertex:')
#             # print(vertex)
#             # print('\nreward:')
#             # print(reward)
#             # print('\nnext_data.x:')
#             # print(next_data.x)
#             # print('\nnext_global_features:')
#             # print(next_global_features)
#             # print('\ndone:')
#             # print(int(done))
#             # input('press enter')

#             current_graph = next_graph
#             # if get_trajectory:
#             #     current_data = next_data
#             #     current_global_features = next_global_features

#         episode_rewards_all_graphs.append(ep_score)
#         colours_used_all_graphs.append(current_graph['no_colours_used'])

#         # if get_trajectory:
#         #     stats = {
#         #         "name": graph_target['name'],
#         #         "episode_reward": ep_score,
#         #         "colours_used": current_graph['no_colours_used'], 
#         #         "trajectory": trajectory
#         #     }
#         # else:
#         #     stats = {
#         #         "name": graph_target['name'],
#         #         "episode_reward": ep_score,
#         #         "colours_used": current_graph['no_colours_used'],
#         #     }
#         stats = {
#                 "name": graph_target['name'],
#                 "episode_reward": ep_score,
#                 "colours_used": current_graph['no_colours_used'], 
#                 "vertex_order": vertex_order
#             }
        
#         stats_by_graph.append(stats)

#     stats_all_graphs = {
#         "avg_episode_reward": np.mean(episode_rewards_all_graphs),
#         "episode_reward_std": np.std(episode_rewards_all_graphs),
#         "avg_colours_used": np.mean(colours_used_all_graphs),
#         "colours_used_std": np.std(colours_used_all_graphs)
#     }

#     return stats_all_graphs, stats_by_graph


def test_using_learned_policy(graphs, agent, stochastic):
    
    if stochastic:
        runs_per_graph = 100
    else:
        runs_per_graph = 1
    
    test_env = gym.make('graph-colouring', dataset_graphs=graphs)

    stats_by_graph = []
    episode_rewards_all_graphs = []
    colours_used_all_graphs = []
    max_episode_rewards_by_graph = []
    min_colours_used_by_graph = []

    for graph_target in graphs:
        
        episode_rewards=[]
        colours_used = []
        max_episode_reward = -np.inf
        min_colours_used = np.inf

        for repeat in range(runs_per_graph):
            ep_step = 0
            ep_ended = False
            ep_score = 0

            graph_target, current_graph = test_env.reset(target=graph_target)
            current_data, current_global_features = test_env.GenerateData_v1(graph_target, current_graph)
            
            while not ep_ended:

                if ep_step==0:
                    # # option 1: Choose first vertex randomly
                    # vertex_index = random.choice(current_graph['unassigned_vertices'])
                    
                    # option 2: Choose first vertex according to:
                    # - max degree (distance 1 neighbours)
                    # - max no of distance 2 neighbours
                    # - max no of distance 3 neighbours
                    multiplier = np.array([[graph_target['no_vertices']**(MAX_RADIUS_FOR_NEIGHBOURS-1-i) for i in range(MAX_RADIUS_FOR_NEIGHBOURS)] for _ in range(graph_target['no_vertices'])])
                    vertex_index = np.argmax(np.sum(graph_target['neighbour_dist_counts'] * multiplier, axis=1))

                    # # option 3: Choose first vertex using NN:
                    # vertex_index = agent.choose_action(current_data, current_global_features, current_graph['unassigned_vertices'], test_mode=True)
                    
                else:
                    if stochastic:
                        vertex_index = agent.choose_action_stochastic_2(current_data, current_global_features, current_graph['unassigned_vertices'], epsilon=0)
                    else:
                        vertex_index = agent.choose_action(current_data, current_global_features, current_graph['unassigned_vertices'], test_mode=True)

                next_graph, reward, done, info = test_env.step(vertex_index)
                next_data, next_global_features = test_env.GenerateData_v1(graph_target, next_graph)
                
                current_graph = next_graph
                current_data = next_data
                current_global_features = next_global_features

                if 'episode_ended' in info.keys():
                    ep_ended = info['episode_ended']
                ep_ended = ep_ended or done

                ep_score += reward
                ep_step += 1

            episode_rewards.append(ep_score)
            colours_used.append(current_graph['no_colours_used'])
            max_episode_reward = max(max_episode_reward, ep_score)
            min_colours_used = min(min_colours_used, current_graph['no_colours_used'])

        # stats_for_graph = namedtuple("Stats",["episode_rewards", "avg_episode_reward", "episode_reward_std", "colours_used", "avg_colours_used", "colours_used_std"])(
        #     episode_rewards=episode_rewards,
        #     avg_episode_reward=np.mean(episode_rewards),
        #     episode_reward_std=np.std(episode_rewards),
        #     colours_used = colours_used,
        #     avg_colours_used=np.mean(colours_used),
        #     colours_used_std=np.std(colours_used)
        # )

        episode_rewards_all_graphs += episode_rewards
        colours_used_all_graphs += colours_used
        max_episode_rewards_by_graph.append(max_episode_reward)
        min_colours_used_by_graph.append(min_colours_used)

        stats_for_graph = {
            "name": graph_target['name'],
            "avg_episode_reward": np.mean(episode_rewards),
            "episode_reward_std": np.std(episode_rewards),
            "max_episode_reward": max_episode_reward,
            "avg_colours_used": np.mean(colours_used),
            "colours_used_std": np.std(colours_used),
            "min_colours_used": min_colours_used
        }

        stats_by_graph.append(stats_for_graph)

    stats_all_graphs = {
        "avg_max_episode_reward": np.mean(max_episode_rewards_by_graph),
        "avg_episode_reward": np.mean(episode_rewards_all_graphs),
        "episode_reward_std": np.std(episode_rewards_all_graphs),
        "avg_min_colours_used": np.mean(min_colours_used_by_graph),
        "avg_colours_used": np.mean(colours_used_all_graphs),
        "colours_used_std": np.std(colours_used_all_graphs)
    }

    return stats_all_graphs, stats_by_graph





# # def test_on_validation_graphs(test_env, agent, validation_graphs, eps_per_graph=1, saved_params_filepath=None, epsilon=0):

# #     # first_graph_no_vertices, first_graph_target_tuple = validation_graphs[0]
    
# #     # _, example_current_graph = test_env.reset(no_vertices_for_ep=first_graph_no_vertices, target_tuple=first_graph_target_tuple)

# #     # example_data = test_env.GenerateData_v1(test_env.target, example_current_graph)
# #     # len_node_features = example_data.x.shape[1]
# #     # len_edge_features = example_data.edge_attr.shape[1]


# #     if not saved_params_filepath is None:
# #         agent.load_models(saved_params_filepath_root)

# #     stats = namedtuple("Stats",["episode_rewards", "colours_used"])(
# #         episode_rewards=[],
# #         colours_used = []
# #         )

# #     no_test_episodes = len(validation_graphs) * eps_per_graph

# #     for validation_graph_target in validation_graphs:
        
# #         # print('\n**************************')
# #         # print('no_vertices:')
# #         # print(no_vertices)
# #         # print('edge_list:')
# #         # print(graph_tuple[0])
# #         # print('edge_attr:')
# #         # print(graph_tuple[1])
# #         # print('edge_ind_dict:')
# #         # print(graph_tuple[2])
# #         # print('target_edge_indices:')
# #         # print(graph_tuple[3])
# #         for repeat in range(eps_per_graph):
# #             ep_step = 0
# #             ep_ended = False
# #             ep_score = 0

# #             target, current_graph = test_env.reset(target=validation_graph_target)
            
# #             current_data, current_global_features = test_env.GenerateData_v1(target, current_graph)
# #             while not ep_ended:
# #                 if ep_step <=1:
# #                     flag=True
# #                 else:
# #                     flag=False
# #                 if epsilon==1:
# #                     vertex_index = choose_random_action(current_graph['unassigned_vertices'])
# #                 else:
# #                     if ep_step==0:
# #                         # if all the vertices are uncoloured, choose the vertex according to:
# #                         # - max degree (distance 1 neighbours)
# #                         # - max no of distance 2 neighbours
# #                         # - max no of distance 3 neighbours
# #                         multiplier = np.array([[target['no_vertices']**(MAX_RADIUS_FOR_NEIGHBOURS-1-i) for i in range(MAX_RADIUS_FOR_NEIGHBOURS)] for _ in range(target['no_vertices'])])
# #                         vertex_index = np.argmax(np.sum(target['neighbour_dist_counts'] * multiplier, axis=1))
# #                     else:
# #                         vertex_index = agent.choose_action(current_data, current_global_features, current_graph['unassigned_vertices'], epsilon=epsilon, flag=flag)

# #                 next_graph, reward, done, info = test_env.step(vertex_index)
                
# #                 next_data, next_global_features = test_env.GenerateData_v1(target, next_graph)
                
# #                 current_graph = next_graph
# #                 current_data = next_data
# #                 current_global_features = next_global_features

# #                 if 'episode_ended' in info.keys():
# #                     ep_ended = info['episode_ended']
# #                 ep_ended = ep_ended or done

# #                 ep_score += reward
# #                 ep_step += 1

# #             stats.episode_rewards.append(ep_score)
# #             stats.colours_used.append(current_graph['no_colours_used'])
        
# #     return stats





# # if __name__ == "__main__":
# #     MIN_NO_VERTICES = 25
# #     MAX_NO_VERTICES = 25

# #     MIN_NO_COLOURS = 5
# #     MAX_NO_COLOURS = 5
# #     env = gym.make('graph-colouring-ultd-v1', min_no_vertices=MIN_NO_VERTICES, max_no_vertices=MAX_NO_VERTICES, min_no_colours=MIN_NO_COLOURS, max_no_colours=MAX_NO_COLOURS)
# #     validation_graphs = env.validation_graphs

# #     test_on_validation_graphs(env.validation_graphs, eps_per_graph=1, saved_params_filepath=None, epsilon=1)


# # if __name__ == "__main__":
# #     GRAPH_FILENAME = 'lemos/queen8_8.col'
# #     full_graph_filepath = os.path.join('colouring_graphs', GRAPH_FILENAME)
# #     no_orig_vertices, target, edge_ind_dict = import_dimacs_graph(full_graph_filepath)

# #     import gym_graph_colouring_ultd_v1
# #     env = gym.make('graph-colouring-ultd-v1')

# #     TRAINING_DIRNAME = '20220708_1145_gn'
# #     TRAINED_PARAMS_EPISODE = 5000
# #     training_run_base_dir = os.path.join('outputs', 'training', TRAINING_DIRNAME)
# #     saved_params_filepath_root = os.path.join(training_run_base_dir, 'trained_params', 'episode_' + str(TRAINED_PARAMS_EPISODE))

# #     _, example_current_graph = env.reset(no_vertices_for_ep=no_orig_vertices, target_tuple=(target['edge_list'], target['edge_attr'], edge_ind_dict, target['target_edge_indices']))
# #     example_data = env.GenerateData_v1(target, example_current_graph)
# #     len_node_features = example_data.x.shape[1]
# #     len_edge_features = example_data.edge_attr.shape[1]

# #     validation_graphs = [(no_orig_vertices, (target['edge_list'], target['edge_attr'], edge_ind_dict, target['target_edge_indices']))]
# #     agent = DQNAgentGN(env, len_node_features, len_edge_features, validation_graphs)

# #     agent.load_models(saved_params_filepath_root)
    
# #     validation_stats = test_on_validation_graphs(env, agent, validation_graphs, eps_per_graph=5, epsilon=0)
# #     print('\n==============================================')
# #     print('Current performance on validation graphs:')
# #     print('Average reward: %.2f' % np.mean(validation_stats.episode_rewards), '| Average colours used: %.2f' % np.mean(validation_stats.colours_used))
# #     print('==============================================\n')    

    

