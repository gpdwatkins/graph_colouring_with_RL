import gym
import numpy as np
from utils import MAX_RADIUS_FOR_NEIGHBOURS


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

