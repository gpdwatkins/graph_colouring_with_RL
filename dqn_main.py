import numpy as np
import random
from datetime import datetime
# import pickle
import os
import time
from collections import namedtuple
import gym
import gym_graph_colouring
from dqn_agent import DQNAgentGN
from utils import save_benchmark_stats_to_file, save_training_stats_to_file, save_validation_stats_to_file
from dataset_functions import *
import torch
# from torch_geometric.data import Data, Batch
import cProfile
import pstats
import wandb

PROFILE = False
SAVE_OUTPUTS = False
USE_WANDB = False

# Set the seeds to use for the experiments. One policy will be learned per seed
manual_seeds = [0]
# Name each of the training runs (one name per seed)
exp_names = ['exp0']
# Write a description for each of the training runs (one per seed)
training_run_readmes = [
    'Standard ReLCol algorithm, seed=0'
    ]

for manual_seed, exp_name, training_run_readme in zip(manual_seeds, exp_names, training_run_readmes):

    print('\nStarting new run with readme:')
    print(training_run_readme)
    print('\n')

    if USE_WANDB:
        wandb.init(project="graph_colouring", config={})

    if PROFILE:
        print('Profiling this run')
        profile = cProfile.Profile()
        profile.enable()

    if SAVE_OUTPUTS:
        start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.join('outputs', 'training', start_time + '_gn')
        print('\nSaving output to directory', base_dir)
        os.mkdir(base_dir)

    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # load the graphs for training/validation
    training_dataset = load_dataset(os.path.join('datasets', 'training_dataset.pickle'))
    validation_dataset = load_dataset(os.path.join('datasets', 'validation_dataset.pickle'))

    if SAVE_OUTPUTS:
        readme_filename = os.path.join(base_dir, 'readme.txt')
        with open(readme_filename, 'w') as file:
            file.write(training_run_readme)

        stats_dir = os.path.join(base_dir, 'training_analysis')
        os.mkdir(stats_dir)
        trained_params_dir = os.path.join(base_dir, 'trained_params')
        os.mkdir(trained_params_dir)

    # create the environment
    env = gym.make('graph-colouring-v0', dataset_graphs=training_dataset.graphs)
    
    # start a 'dummy' episode to get an example state
    example_target, example_current_graph = env.reset()
    example_data, example_global_features = env.GenerateData_v1(example_target, example_current_graph)
    len_node_features = example_data.x.shape[1]
    len_edge_features = example_data.edge_attr.shape[1]
    len_global_features = len(example_global_features)

    # create the agent
    agent = DQNAgentGN(len_node_features, len_edge_features, len_global_features)

    training_stats = namedtuple("Stats",["saved_episodes", "episode_rewards", "colours_used"])(
        saved_episodes=[],
        episode_rewards=[],
        colours_used = []
        )

    all_validation_stats = {}
    best_validation_reward = -np.inf
    best_validation_filepath = None

    # ##############################################################
    # print benchmark stats, including average performance of random policy and DSATUR
    print('\n=========================================')
    print('Benchmark performance with random policy:')
    print('Average reward: %.2f' % validation_dataset.random_policy_stats_combined['avg_episode_reward'], '| Average colours used: %.2f' % validation_dataset.random_policy_stats_combined['avg_colours_used'])
    print('DSATUR reward: %.2f' % validation_dataset.dsatur_stats_combined['avg_episode_reward'], '| Average colours used: %.2f' % validation_dataset.dsatur_stats_combined['avg_colours_used'])
    print('=========================================\n')
    # ##############################################################

    if USE_WANDB:
        wandb.config.update({
            "seed": manual_seed,
            "exp_name": exp_name,
            "directory": base_dir,
            "gamma": agent.gamma,
            "alpha": agent.alpha,
            "tau": agent.tau,
            "batch_size": agent.batch_size,
            "no_episodes": agent.no_episodes,
            "random policy benchmark colours used": validation_dataset.random_policy_stats_combined['avg_colours_used'],
            "DSATUR benchmark colours used": validation_dataset.dsatur_stats_combined['avg_colours_used']
            }
        )

    benchmark_stats = namedtuple("Stats",["episode_rewards", "colours_used"])(
        episode_rewards=[validation_dataset.random_policy_stats_by_graph[i]['avg_episode_reward'] for i in range(len(validation_dataset.graphs))],
        colours_used = [validation_dataset.random_policy_stats_by_graph[i]['avg_colours_used'] for i in range(len(validation_dataset.graphs))]
        )

    t_step = 0
    ep_ind=0

    # do an initial run of the untrained policy on the validation dataset    
    validation_stats, validation_stats_by_graph = run_learned_policy_on_dataset(validation_dataset.graphs, agent)
            
    ep_validation_stats = namedtuple("Stats",["episode_rewards", "colours_used"])(
        episode_rewards=[validation_stats_by_graph[i]['avg_episode_reward'] for i in range(len(validation_dataset.graphs))],
        colours_used = [validation_stats_by_graph[i]['avg_colours_used'] for i in range(len(validation_dataset.graphs))],
        )
    all_validation_stats[ep_ind] = ep_validation_stats

    print('==============================================')
    # print performance of untrained policy on validation set
    print('Current performance on validation graphs:')
    print('Average reward: %.2f' % validation_stats['avg_episode_reward'], '| Average colours used: %.2f' % validation_stats['avg_colours_used'])
    print('==============================================\n')

    if USE_WANDB:
        wandb.log({"validation avg ep reward": validation_stats['avg_episode_reward']})
        wandb.log({"validation avg colours used": validation_stats['avg_colours_used']})

    if (validation_stats['avg_episode_reward'] >= best_validation_reward):
        if SAVE_OUTPUTS:
            if not best_validation_filepath is None:
                os.unlink(best_validation_filepath)
            
            best_validation_reward = validation_stats['avg_episode_reward']
            best_validation_filepath = os.path.join(trained_params_dir, 'best_policy_episode_' + str(ep_ind))
            
            best_validation_filepath = agent.save_models(best_validation_filepath)

        if USE_WANDB:
            wandb.run.summary["best validation avg ep reward"] = validation_stats['avg_episode_reward']
            wandb.run.summary["best validation avg colours used"] = validation_stats['avg_colours_used']

    for ep_ind in range(1, agent.no_episodes+1):
        
        ep_step = 0
        ep_ended = False
        ep_score = 0

        # start a new episode
        target, current_graph = env.reset()
        current_data, current_global_features = env.GenerateData_v1(target, current_graph)

        while not ep_ended:
            
            # given the current state, choose an action
            if ep_step==0:
                # first vertex is chosen at random
                first_vertex_uses_NN = False
                vertex_index = random.choice(current_graph['unassigned_vertices'])
            else:
                vertex_index = agent.choose_action(current_data, current_global_features, current_graph['unassigned_vertices'])

            # submit action to the environment and observe next state, reward and done
            next_graph, reward, done, info = env.step(vertex_index)
            next_data, next_global_features = env.GenerateData_v1(target, next_graph)

            if first_vertex_uses_NN or ep_step>0:
                # don't want to learn from transitions that weren't chosen using neural network
                agent.remember(current_data, current_global_features, np.array(next_graph['assigned_vertices_onehot']), vertex_index, reward, next_data, next_global_features, int(done))

            # every update_every steps, update the networks
            if t_step%agent.update_every == 0:
                agent.learn()
            t_step += 1

            current_data = next_data
            current_global_features = next_global_features

            if 'episode_ended' in info.keys():
                ep_ended = info['episode_ended']
            ep_ended = ep_ended or done

            ep_score += reward
            ep_step += 1

        training_stats.saved_episodes.append(ep_ind)
        training_stats.episode_rewards.append(ep_score)
        training_stats.colours_used.append(env.current_graph['no_colours_used'])

        if ep_ind>0 and ep_ind%100 == 0:
            # periodically print training stats
            print('#####################################')
            average_rewards = np.mean(training_stats.episode_rewards[-100:])
            average_colour_used = np.mean(training_stats.colours_used[-100:])
            print('episode: ', ep_ind, '| ep score: %.2f' % ep_score, '| 100 game average reward: %.2f' % average_rewards, '| ep colours used: %.2f' % env.current_graph['no_colours_used'], '| 100 game average colours_used: %.2f' % average_colour_used)
            if USE_WANDB:
                wandb.log({"training episode reward": average_rewards})
                wandb.log({"training colours used": average_colour_used})
        
        if ep_ind%500 == 0:
            # periodically run current policy on validation set
            validation_stats, validation_stats_by_graph = run_learned_policy_on_dataset(validation_dataset.graphs, agent)
            
            ep_validation_stats = namedtuple("Stats",["episode_rewards", "colours_used"])(
                episode_rewards=[validation_stats_by_graph[i]['avg_episode_reward'] for i in range(len(validation_dataset.graphs))],
                colours_used = [validation_stats_by_graph[i]['avg_colours_used'] for i in range(len(validation_dataset.graphs))],
                )

            all_validation_stats[ep_ind] = ep_validation_stats

            print('\n==============================================')
            # print performance of current policy on validation set
            print('Current performance on validation graphs:')
            # print('Average reward: %.2f' % np.mean(validation_stats.episode_rewards), '| Average colours used: %.2f' % np.mean(validation_stats.colours_used))
            print('Average reward: %.2f' % validation_stats['avg_episode_reward'], '| Average colours used: %.2f' % validation_stats['avg_colours_used'])
            print('==============================================\n')

            if USE_WANDB:
                wandb.log({"validation avg ep reward": validation_stats['avg_episode_reward']})
                wandb.log({"validation avg colours used": validation_stats['avg_colours_used']})

            if (validation_stats['avg_episode_reward'] >= best_validation_reward):
                if SAVE_OUTPUTS:
                    if not best_validation_filepath is None:
                        os.unlink(best_validation_filepath)

                    best_validation_reward = validation_stats['avg_episode_reward']
                    best_validation_filepath = os.path.join(trained_params_dir, 'best_policy_episode_' + str(ep_ind))

                    best_validation_filepath = agent.save_models(best_validation_filepath)

                if USE_WANDB:
                    wandb.run.summary["best validation avg ep reward"] = validation_stats['avg_episode_reward']
                    wandb.run.summary["best validation avg colours used"] = validation_stats['avg_colours_used']

        if SAVE_OUTPUTS:
            if ep_ind == agent.no_episodes:
                # Save trained model parameters 
                params_checkpoint_filepath_root = os.path.join(trained_params_dir, 'final_policy_episode_' + str(ep_ind))
                _ = agent.save_models(params_checkpoint_filepath_root)
            
            if ep_ind>0 and ((ep_ind%1000 == 0) or (ep_ind == agent.no_episodes)):
                # Save training stats
                for file in os.listdir(stats_dir):
                    file_with_path = os.path.join(stats_dir, file)
                    if os.path.isfile(file_with_path):
                        os.unlink(file_with_path)
                benchmark_stats_filename = os.path.join(stats_dir, 'benchmark_stats')
                save_benchmark_stats_to_file(benchmark_stats, benchmark_stats_filename)

                training_stats_filename = os.path.join(stats_dir, 'training_stats_episode_' + str(ep_ind) + '.txt')
                save_training_stats_to_file(training_stats, training_stats_filename)

                validation_stats_filename = os.path.join(stats_dir, 'validation_stats_episode_' + str(ep_ind) + '.txt')
                save_validation_stats_to_file(all_validation_stats, validation_stats_filename)
        
        agent.episode_no += 1

    if USE_WANDB:
        time.sleep(5)
        wandb.finish()
        time.sleep(25)

    if PROFILE:
        profile.disable()
        ps = pstats.Stats(profile)
        ps.sort_stats('cumtime', 'calls') 
        ps.print_stats(50)