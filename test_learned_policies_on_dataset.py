import pickle
import os
import glob

import gym_graph_colouring

from dataset_functions import *
from test_policy_functions import *
from dqn_agent import DQNAgentGN

PRINT_OUTPUT = True
SAVE_OUTPUT = False

# A string of the filenames of the trained parameters, saved in outputs/training
TRAINING_DIRNAMES = ['learned_parameters_GN']



DATASET_FILENAME = 'lemos_20221026_1355.pickle'
dataset_filepath = os.path.join('datasets', DATASET_FILENAME)

# Note runs per graphs is fixed in test_using_learned_policy function (in file test_policy_functions.py)

TEST_STOCHASTIC_POLICY = False

def get_multiple_policy_stats(dataset, all_stats):
    graph_names=[]
    for target_graph in dataset.graphs:
        graph_names.append(target_graph['name'])
    
    colours_used_by_graph = {}
    for graph_name in graph_names:
        colours_used_by_graph[graph_name] = np.zeros(len(TRAINING_DIRNAMES))

    for ind, training_run_stats in enumerate(all_stats):
        if TEST_STOCHASTIC_POLICY:
            deterministic_stats_all_graphs, deterministic_stats_by_graph, stochastic_stats_all_graphs, stochastic_stats_by_graph = training_run_stats
        else:
            deterministic_stats_all_graphs, deterministic_stats_by_graph = training_run_stats
    
        for individual_graph_stats in deterministic_stats_by_graph:
            colours_used_by_graph[individual_graph_stats['name']][ind] = individual_graph_stats['avg_colours_used']

    stats_by_graph = {}
    for graph_name in graph_names:
        stats_by_graph[graph_name] = {}
        stats_by_graph[graph_name]['mean'] = np.mean(colours_used_by_graph[graph_name])
        stats_by_graph[graph_name]['std'] = np.std(colours_used_by_graph[graph_name])
        stats_by_graph[graph_name]['min'] = np.amin(colours_used_by_graph[graph_name])
        stats_by_graph[graph_name]['max'] = np.amax(colours_used_by_graph[graph_name])

    return graph_names, stats_by_graph

def print_baselines(dataset):

    print('Dataset results')
    print('===============')
    print('\n')
    
    print('Random policy avg reward, avg colours and std of colours used by graph: ')
    for graph_stats in dataset.random_policy_stats_by_graph:
        print(graph_stats['name'], ': ', graph_stats['avg_episode_reward'], ' | ', graph_stats['avg_colours_used'], ' | ', graph_stats['colours_used_std'])
    print('On average:')
    print(dataset.random_policy_stats_combined['avg_episode_reward'], ' | ', dataset.random_policy_stats_combined['avg_colours_used'])
    
    print('---')
    print('DSATUR avg reward and avg colours used by graph: ')
    for graph_stats in dataset.dsatur_stats_by_graph:
        print(graph_stats['name'], ': ', graph_stats['episode_reward'], ' | ', graph_stats['colours_used'])
    print('On average:')
    print(dataset.dsatur_stats_combined['avg_episode_reward'], ' | ', dataset.dsatur_stats_combined['avg_colours_used'])


def print_single_policy_results(dataset, all_stats):
    
    print_baselines(dataset)

    if TEST_STOCHASTIC_POLICY:
        deterministic_stats_all_graphs, deterministic_stats_by_graph, stochastic_stats_all_graphs, stochastic_stats_by_graph = all_stats
    else:
        deterministic_stats_all_graphs, deterministic_stats_by_graph = all_stats
    
    print('---')
    print('Learned policy (deterministic) avg reward and avg colours used: ')
    for graph_stats in deterministic_stats_by_graph:
        print(graph_stats['name'], ': ', graph_stats['avg_episode_reward'], ' | ', graph_stats['avg_colours_used'])
    print('On average:')
    print(deterministic_stats_all_graphs['avg_episode_reward'], ' | ', deterministic_stats_all_graphs['avg_colours_used'])
    
    if TEST_STOCHASTIC_POLICY:
        print('---')
        print('Learned policy (stochastic) avg reward, avg colours used and min colours used: ')
        for graph_stats in stochastic_stats_by_graph:
            print(graph_stats['name'], ': ', graph_stats['avg_episode_reward'], ' | ', graph_stats['avg_colours_used'], ' | ', graph_stats['min_colours_used'])
        print('On average:')
        print(stochastic_stats_all_graphs['avg_episode_reward'], ' | ', stochastic_stats_all_graphs['avg_colours_used'], ' | ', stochastic_stats_all_graphs['avg_min_colours_used'])


def print_multiple_policy_results(graph_names, dataset, stats_by_training_run, stats_by_graph):
    
    print_baselines(dataset)

    print('---')
    print('Learned policy (deterministic) colours used stats by run across graphs:')
    print(f"{'run_no':<10}{'|'}{'mean' :<6}{'|'}{'std':<6}")
    print('-'*(10+6*2))
    run_means = np.zeros(len(stats_by_training_run))
    for ind, elt in enumerate(stats_by_training_run):
        run_means[ind] = elt[0]['avg_colours_used']
        print(f"""{ind:<10}{'|'}{f'''{elt[0]['avg_colours_used']:.2f}'''.rstrip('0').rstrip('.'):<6}{'|'}{f'''{elt[0]['colours_used_std']:.2f}'''.rstrip('0').rstrip('.'):<6}""")
    print('-'*(10+6*2))
    print('On average: ', np.mean(run_means))
    print('With std: ', np.std(run_means))
    
    print('---')
    print('Learned policy (deterministic) colours used stats by graph across runs:')
    print(f"{'graph_name':<20}{'|'}{'mean' :<6}{'|'}{'std':<6}{'|'}{'min':<6}{'|'}{'max':<6}")
    print('-'*(20+6*4))
    graph_means = np.zeros(len(graph_names))
    for ind, graph_name in enumerate(graph_names):
        graph_means[ind] = stats_by_graph[graph_name]['mean']
        print(f"""{graph_name:<20}{'|'}{f'''{stats_by_graph[graph_name]['mean']:.2f}'''.rstrip('0').rstrip('.'):<6}{'|'}{f'''{stats_by_graph[graph_name]['std']:.2f}'''.rstrip('0').rstrip('.'):<6}{'|'}{stats_by_graph[graph_name]['min']:<6.0f}{'|'}{stats_by_graph[graph_name]['max']:<6.0f}""")
    print('-'*(20+6*4))
    print('On average: ', np.mean(graph_means))
    print('With std: ', np.std(graph_means))


if __name__ == "__main__":
    
    print('\n=================================================')
    testing_run_readme = input('Please write a description for this testing run: \n')
    print('=================================================\n')
    
    if SAVE_OUTPUT:
        start_time = datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = os.path.join('outputs', 'testing', 'dataset_tests', start_time)
        print('Saving output to directory', output_dir)
        os.mkdir(output_dir)

        readme_filename = os.path.join(output_dir, 'readme.txt')
        with open(readme_filename, 'a') as file:
            file.write('Dataset: ' + DATASET_FILENAME + '\n')
            file.write('Policies: ' + ', '.join(TRAINING_DIRNAMES) + '\n')
            file.write('Using ' + BEST_OR_FINAL_POLICY + ' policy\n\n')
            file.write(testing_run_readme)
    
    env = gym.make('graph-colouring-v0')

    dataset = load_dataset(dataset_filepath)

    example_target, example_current_graph = env.reset(target=dataset.graphs[0])

    example_data, example_global_features = env.GenerateData_v1(example_target, example_current_graph)
    len_node_features = example_data.x.shape[1]
    len_edge_features = example_data.edge_attr.shape[1]
    len_global_features = len(example_global_features)

    agent = DQNAgentGN(len_node_features, len_edge_features, len_global_features, avg_deg=dataset.avg_deg, test_mode=True)

    stats_by_training_run = []
    for training_dirname in TRAINING_DIRNAMES:

        training_run_base_dir = os.path.join('trained_policies')
        print('os.path.join(training_run_base_dir, training_dirname):')
        print(os.path.join(training_run_base_dir, training_dirname))
        saved_params_filepath_root = glob.glob(os.path.join(training_run_base_dir, training_dirname))

        if len(saved_params_filepath_root) != 1:
            raise Exception('Should have only found one saved params file')
        else:
            saved_params_filepath_root = saved_params_filepath_root[0]

        agent.load_models(saved_params_filepath_root)

        deterministic_stats_all_graphs, deterministic_stats_by_graph = test_using_learned_policy(dataset.graphs, agent, stochastic=False)
        if TEST_STOCHASTIC_POLICY:
            stochastic_stats_all_graphs, stochastic_stats_by_graph = test_using_learned_policy(dataset.graphs, agent, stochastic=True)
            stats_by_training_run.append((deterministic_stats_all_graphs, deterministic_stats_by_graph, stochastic_stats_all_graphs, stochastic_stats_by_graph))
        else:
            stats_by_training_run.append((deterministic_stats_all_graphs, deterministic_stats_by_graph))

    if len(TRAINING_DIRNAMES) != 1:
        graph_names, stats_by_graph = get_multiple_policy_stats(dataset, stats_by_training_run)
        if SAVE_OUTPUT:
            output_stats_filepath = os.path.join(output_dir, 'colouring_stats.pickle')
            pickle.dump(stats_by_graph, open(output_stats_filepath, "wb" ))
    
    if PRINT_OUTPUT:
        if len(TRAINING_DIRNAMES) == 1:
            print_single_policy_results(dataset, stats_by_training_run[0])
        else:            
            print_multiple_policy_results(graph_names, dataset, stats_by_training_run, stats_by_graph)
        
        