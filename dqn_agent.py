import numpy as np
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from architecture import GNNetwork

class ReplayBuffer(object):
    def __init__(self, max_size, global_features_size):
        """
        Initialise the replay buffer for storing transitions.
        Parameters
        ==========
            max_size: The number of transitions the buffer can hold
            global_features_size: The dimension of the global features

        """
        self.max_mem_size = max_size
        self.current_mem_size = 0

        self.data_memory = [None for _ in range(self.max_mem_size)]
        self.current_global_features = np.zeros((self.max_mem_size, global_features_size), dtype=np.float32)
        self.assigned_vertices_memory = np.zeros(self.max_mem_size, dtype=object)
        self.action_ind_memory = np.zeros((self.max_mem_size, 1))
        self.reward_memory = np.zeros(self.max_mem_size)
        self.next_data_memory = [None for _ in range(self.max_mem_size)]
        self.next_global_features = np.zeros((self.max_mem_size, global_features_size), dtype=np.float32)
        self.done_memory = np.zeros(self.max_mem_size, dtype=np.intc)


    def store_transition(self, data, current_global_features, assigned_vertices, action_ind, reward, next_data, next_global_features, done):
        # save a transition to the buffer
        index = self.current_mem_size % self.max_mem_size
        self.data_memory[index] = data.cpu()
        self.current_global_features[index] = current_global_features
        self.assigned_vertices_memory[index] = assigned_vertices
        self.action_ind_memory[index] = action_ind
        self.reward_memory[index] = reward
        self.next_data_memory[index] = next_data.cpu()
        self.next_global_features[index] = next_global_features
        self.done_memory[index] = done
        self.current_mem_size += 1


    def sample_buffer(self, sample_size):
        # randomly sample a set of transitions from the buffer
        max_mem = min(self.current_mem_size, self.max_mem_size)
        batch = np.random.choice(max_mem, sample_size)

        datas_list = []
        next_datas_list = []
        for batch_ind in batch:
            datas_list.append(self.data_memory[batch_ind])
            next_datas_list.append(self.next_data_memory[batch_ind])
        datas_batch = Batch.from_data_list(datas_list)
        next_datas_batch = Batch.from_data_list(next_datas_list)
        
        current_global_features = self.current_global_features[batch]
        assigned_vertices = self.assigned_vertices_memory[batch]
        action_inds = self.action_ind_memory[batch]
        rewards = self.reward_memory[batch]
        next_global_features = self.next_global_features[batch]
        dones = self.done_memory[batch]

        return datas_batch, current_global_features, assigned_vertices, action_inds, rewards, next_datas_batch, next_global_features, dones


class DQNAgentGN():
    def __init__(self, len_vertex_features, len_edge_features, len_global_features, test_mode=False):
        """Initialize the agent.
        The agent chooses actions based on its policy, which it improves using the reward signal.
        Params
        ======
            len_vertex_features: Dimension of vertex features
            len_edge_features: Dimension of edge features
            len_global_features: Dimension of graph-level features
            test_mode: Indicates whether agent should update policy
        """
        # episodic so set discount factor = 1
        self.gamma = 1
        self.alpha = 1e-3
        self.tau = 1e-3
        self.no_episodes = 25000
        self.eps_start = .9
        self.eps_end = .01
        self.max_buffer_size = int(1e5)
        self.batch_size = 64
        self.update_every = 5

        self.len_global_features = len_global_features

        self.learn_counter = 0

        # Initialise behaviour network
        self.gn_behaviour = GNNetwork(len_vertex_features, len_edge_features, len_global_features, name = 'GN', alpha = self.alpha)
        
        if not test_mode:
            # only need target network if the policy is being trained
            self.gn_target = GNNetwork(len_vertex_features, len_edge_features, len_global_features, name = 'GN_target', alpha = self.alpha)
            self.soft_update(self.gn_behaviour, self.gn_target, 1)
            
            # Initialise replay buffer
            self.memory = ReplayBuffer(self.max_buffer_size, len_global_features)

        self.episode_no = 0

    
    def choose_action(self, data, global_features, unassigned_vertices, epsilon=None, test_mode = False):
        # This function takes the current state and returns an action to perform
        # The state is described by the data, global_features and unassigned_vertices inputs

        # during training, follow ε-greedy policy, but at test time use ε=0
        if epsilon is None:
            if test_mode:
                epsilon=0
            else:
                epsilon_decay = (self.eps_end/self.eps_start)**(1/self.no_episodes)
                epsilon = self.eps_start * (epsilon_decay**self.episode_no)

        if random.random() > epsilon:
            # use GNN to select next action
            gn_input = data.to(self.gn_behaviour.device)

            global_features_tensor = torch.tensor(global_features, requires_grad=False, dtype=torch.float).unsqueeze(0).to(self.gn_behaviour.device)
            vertex_batch_map = [0] * data.x.shape[0]
            edge_batch_map = [0] * data.edge_attr.shape[0]

            with torch.no_grad():
                vertex_embeddings, action_values = self.gn_behaviour(gn_input, global_features_tensor, vertex_batch_map, edge_batch_map)

            action_values = action_values.view([-1])
            filtered_action_values = action_values[unassigned_vertices].cpu().data.numpy()
            unassigned_vertex_ind = np.argmax(filtered_action_values)
            vertex_ind = unassigned_vertices[unassigned_vertex_ind]

        else:
            # select next action at random
            vertex_ind = random.choice(unassigned_vertices)

        return vertex_ind


    def remember(self, data, current_global_features, assigned_vertices, action_ind, reward, next_data, next_global_features, done):
        # Add a transition to the replay buffer
        self.memory.store_transition(data, current_global_features, assigned_vertices, action_ind, reward, next_data, next_global_features, done)

    def learn(self):
        # This function updates the GNN if there are a sufficient number of transitions in the replay buffer
        if self.memory.current_mem_size < 10*self.batch_size:
            return

        if self.learn_counter == 0:
            print('Starting learning')
            self.learn_counter += 1

        # Sample batch from buffer
        current_datas_batch, current_global_features, assigned_vertices, action_inds, rewards, next_datas_batch, next_global_features, dones = self.memory.sample_buffer(self.batch_size)
        
        # Move sample to GPU
        current_datas_batch_tensor = current_datas_batch.to(self.gn_behaviour.device)
        current_global_features_tensor = torch.tensor(current_global_features, dtype=torch.float).to(self.gn_behaviour.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(self.gn_behaviour.device)
        next_datas_batch_tensor = next_datas_batch.to(self.gn_behaviour.device)
        next_global_features_tensor = torch.tensor(next_global_features, dtype=torch.float).to(self.gn_behaviour.device)
        dones_tensor = torch.tensor(dones).unsqueeze(1).to(self.gn_behaviour.device)

        no_vertices_per_graph = [len(assigned_vertices_array) for assigned_vertices_array in assigned_vertices]
        
        # need to associate vertices/edges with the graph they belong to as graphs are different sizes
        vertex_to_batch_map = [graph_ind for graph_ind, no_vertices in enumerate(no_vertices_per_graph) for _ in range(no_vertices)]
        edge_to_batch_map = [graph_ind for graph_ind, no_vertices in enumerate(no_vertices_per_graph) for _ in range(no_vertices*(no_vertices-1))]

        # get expected Q values from behaviour network
        _, Q_expected = self.gn_behaviour(current_datas_batch_tensor, current_global_features_tensor, vertex_to_batch_map, edge_to_batch_map, called_by='learn_expected')
        
        # extract Q values for the action taken within each transition in the batch
        # this needs to be done in a loop as batches are modelled as a single large graph 
        Q_expected_aux = torch.zeros((self.batch_size, 1))
        tensor_index=0
        for graph_index, no_vertices in enumerate(no_vertices_per_graph):
            Q_expected_aux[graph_index,0] = Q_expected[tensor_index+action_inds[graph_index]]
            tensor_index += no_vertices

        Q_expected = Q_expected_aux.to(self.gn_behaviour.device) 

        # get target Q values (for next states) from target network
        with torch.no_grad():
            _, Q_targets_next = self.gn_target(next_datas_batch_tensor, next_global_features_tensor, vertex_to_batch_map, edge_to_batch_map, called_by='learn_target')

        Q_targets_next_array = Q_targets_next.cpu().numpy()

        assigned_vertices_array = np.expand_dims(np.concatenate(assigned_vertices), 1)

        # override the Q values for already-coloured vertices to ensure they aren't selected
        Q_targets_next_array[assigned_vertices_array==1] = -1*np.inf

        # extract Q values for the best action from the next state within each transition in the batch
        # this also needs to be done in a loop as batches are modelled as a single large graph 
        Q_targets_next_array_aux = np.zeros((self.batch_size, 1))
        array_index=0
        for graph_index, no_vertices in enumerate(no_vertices_per_graph):
            Q_targets_next_array_aux[graph_index, 0] = np.max(Q_targets_next_array[array_index:array_index+no_vertices])
            array_index += no_vertices
        Q_targets_next_array = Q_targets_next_array_aux

        # override the Q values for 'done' states
        Q_targets_next_array[dones==1] = 0

        with torch.no_grad():
            Q_targets_next = torch.from_numpy(Q_targets_next_array).float().to(self.gn_behaviour.device)
            # calculate Q targets
            Q_targets = (rewards_tensor + (self.gamma * Q_targets_next * (1 - dones_tensor)))

        # calculate loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # backpropagate the loss and perform a gradient descent step
        self.gn_behaviour.optimizer.zero_grad()
        loss.backward()
        self.gn_behaviour.optimizer.step()

        # soft update of target network
        self.soft_update(self.gn_behaviour, self.gn_target, self.tau)


    def soft_update(self, behaviour_network, target_network, tau):
        # Perform a soft update on the weights of the target network
        for target_param, behaviour_param in zip(target_network.parameters(), behaviour_network.parameters()):
            target_param.data.copy_(tau*behaviour_param.data + (1.0-tau)*target_param.data)

    def save_models(self, checkpoint_filepath_root):
        # save weights of behaviour network
        saved_model_filepath = self.gn_behaviour.save_checkpoint(checkpoint_filepath_root)
        return saved_model_filepath

    def load_models(self, checkpoint_filepath_root):
        # load behaviour network weights
        self.gn_behaviour.load_checkpoint(checkpoint_filepath_root)
        