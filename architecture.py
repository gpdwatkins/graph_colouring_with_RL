import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, Size
from torch_geometric.data import Data

from principal_neighbourhood_aggregation import AGGREGATORS, SCALERS

DEVICE = "cuda:4"

class GNNBlock(MessagePassing):
    def __init__(self, message_in_dim, message_hidden_dim, message_out_dim, update_in_dim, update_hidden_dim, update_out_dim, aggregators, scalers):
        """
        Initialise a GNN block, consisting of message, aggregate and update functions.
        Parameters
        ==========
            message_in_dim: Dimension of input to message function
            message_hidden_dim: Dimension of hidden layer of message function
            message_out_dim: Dimension of output from message function
            update_in_dim: Dimension of input to update function
            update_hidden_dim: Dimension of hidden layer of update function
            update_out_dim: Dimension of output from update function
            aggregators: aggregation functions to use in PNA
            scalers: scalers to use in PNA
        """
        super(GNNBlock, self).__init__(aggr=None)

        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]
        
        # create message NN
        self.message_mlp = nn.Sequential(
            nn.Linear(message_in_dim, message_hidden_dim),
            nn.ReLU(),
            nn.Linear(message_hidden_dim, message_out_dim)
            )

        # create update NN
        if not (update_in_dim is None or update_out_dim is None):
            self.update_mlp = nn.Sequential(
                nn.Linear(update_in_dim, update_hidden_dim),
                nn.ReLU(),
                nn.Linear(update_hidden_dim, update_out_dim)
                )

        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def message(self, x_i, x_j, edge_attr):
        # Message function takes in:
        # - vertex features x_i of the source vertex i
        # - vertex features x_j of the destination vertex j
        # - edge features of e_ij
        # and outputs:
        # - edge embedding of e_ij
        message_nn_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        message_nn_output = self.message_mlp(message_nn_input)
        return message_nn_output

    def aggregate(self, inputs, index, edge_attr, dim_dim=None):
        # uses PNA to aggregate embeddings of edges arriving at each vertex
        outs = [aggr(inputs, index, dim_dim) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)
        return out

    def update(self, aggr_out, x):
        # Update function takes in:
        # - Aggregation of edges arriving at vertex i
        # - vertex features x_i of vertex i
        # and outputs:
        # - vertex embedding of i
        if type(x) is tuple:
            x=x[1]
        update_nn_input = torch.cat([aggr_out, x], dim=1)
        update_nn_output = self.update_mlp(update_nn_input)
        return update_nn_output

    def propagate(self, edge_index: Adj, size: Size = None, do_update_step=True, **kwargs):
        # propagate messages around the network
        size = self.__check_input__(edge_index, size)

        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        msg_out = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        aggr_kwargs['edge_attr'] = coll_dict['edge_attr']
        agg_out = self.aggregate(msg_out, **aggr_kwargs)

        update_kwargs = self.inspector.distribute('update', coll_dict)
        upd_out = self.update(agg_out, **update_kwargs)

        return msg_out, upd_out
        
    def forward(self, data):
        return self.propagate(data.edge_index, x=data.x, edge_attr=data.edge_attr)


class GNNetwork(nn.Module):
    def __init__(self, vertex_features_dim, edge_features_dim, global_features_dim, name, alpha):
        """
        Initialise network, consisting of 5 GNN blocks followed by 3 fully-connected layers.
        Parameters
        ==========
            vertex_features_dim: Dimension of the vertex features
            edge_features_dim: Dimension of the edge features
            global_features_dim: Dimension of the graph-level features
            name: Identifier for the network
            alpha: Learning rate
        """
        super(GNNetwork, self).__init__()

        self.name = name

        # aggregators/scalers for PNA
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity']
        
        # Set dimensions of the network
        msg_hidden_dim = 64
        upd_hidden_dim = 64
        msg_output_dim = 64
        upd_output_dim = 64
        fc1_units=64
        fc2_units=64
        actions_per_vertex = 1

        # Calculate further dimensions of the network
        gn1_msg_input_dim = 2*vertex_features_dim + edge_features_dim
        gn1_msg_output_dim = msg_output_dim
        gn1_upd_input_dim = len(aggregators) * len(scalers) * gn1_msg_output_dim + vertex_features_dim
        gn1_upd_output_dim = upd_output_dim
        
        gn2_msg_input_dim = 2*gn1_upd_output_dim + gn1_msg_output_dim
        gn2_msg_output_dim = msg_output_dim
        gn2_upd_input_dim = len(aggregators) * len(scalers) * gn2_msg_output_dim + gn1_upd_output_dim
        gn2_upd_output_dim = upd_output_dim

        gn3_msg_input_dim = 2*gn2_upd_output_dim + gn2_msg_output_dim
        gn3_msg_output_dim = msg_output_dim
        gn3_upd_input_dim = len(aggregators) * len(scalers) * gn3_msg_output_dim + gn2_upd_output_dim
        gn3_upd_output_dim = upd_output_dim

        gn4_msg_input_dim = 2*gn3_upd_output_dim + gn3_msg_output_dim
        gn4_msg_output_dim = msg_output_dim
        gn4_upd_input_dim = len(aggregators) * len(scalers) * gn4_msg_output_dim + gn3_upd_output_dim
        gn4_upd_output_dim = upd_output_dim

        gn5_msg_input_dim = 2*gn4_upd_output_dim + gn4_msg_output_dim
        gn5_msg_output_dim = msg_output_dim
        gn5_upd_input_dim = len(aggregators) * len(scalers) * gn5_msg_output_dim + gn4_upd_output_dim
        gn5_upd_output_dim = upd_output_dim
        
        # Initialise the GNN blocks
        self.gn_layer1 = GNNBlock(gn1_msg_input_dim, msg_hidden_dim, gn1_msg_output_dim, gn1_upd_input_dim, upd_hidden_dim, gn1_upd_output_dim, aggregators, scalers)
        self.gn_layer2 = GNNBlock(gn2_msg_input_dim, msg_hidden_dim, gn2_msg_output_dim, gn2_upd_input_dim, upd_hidden_dim, gn2_upd_output_dim, aggregators, scalers)
        self.gn_layer3 = GNNBlock(gn3_msg_input_dim, msg_hidden_dim, gn3_msg_output_dim, gn3_upd_input_dim, upd_hidden_dim, gn3_upd_output_dim, aggregators, scalers)
        self.gn_layer4 = GNNBlock(gn4_msg_input_dim, msg_hidden_dim, gn4_msg_output_dim, gn4_upd_input_dim, upd_hidden_dim, gn4_upd_output_dim, aggregators, scalers)
        self.gn_layer5 = GNNBlock(gn5_msg_input_dim, msg_hidden_dim, gn5_msg_output_dim, gn5_upd_input_dim, upd_hidden_dim, gn5_upd_output_dim, aggregators, scalers)

        # Initialise the fully-connected layers
        self.fc1 = nn.Linear(gn5_upd_output_dim+global_features_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, actions_per_vertex)

        # Create dummy Data object to avoid in-place modification
        self.dummy_data = Data(x=torch.tensor([]), edge_index = torch.tensor([], dtype=torch.long), edge_attr = torch.tensor([]))

        # Initialise Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        
        # Move network to GPU if available
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, data, global_features, vertex_batch_map, edge_batch_map, called_by=None):
        
        # First pass the data through the GNN blocks and save outputs in dummy_data
        self.dummy_data.edge_index = data.edge_index
        self.dummy_data.edge_attr, self.dummy_data.x = self.gn_layer1(data)
        self.dummy_data.edge_attr, self.dummy_data.x = self.gn_layer2(self.dummy_data)
        self.dummy_data.edge_attr, self.dummy_data.x = self.gn_layer3(self.dummy_data)
        self.dummy_data.edge_attr, self.dummy_data.x = self.gn_layer4(self.dummy_data)
        _, vertex_embeddings = self.gn_layer5(self.dummy_data)

        # Concatenate the vertex embeddings with global features  and save as y
        # Note that if there are no global features, y is equal to vertex_embeddings
        global_features_repeated_for_vertices = global_features[vertex_batch_map]
        y = torch.cat((vertex_embeddings, global_features_repeated_for_vertices), dim=1)

        # Pass y through the fully-connected layers
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        q_values = self.fc3(y)

        return vertex_embeddings, q_values

    def save_checkpoint(self, checkpoint_filepath_root):
        print('... saving checkpoint ...')
        checkpoint_filepath_full = checkpoint_filepath_root + '_' + self.name
        torch.save(self.state_dict(), checkpoint_filepath_full)
        return checkpoint_filepath_full

    def load_checkpoint(self, checkpoint_filepath_root):
        if not checkpoint_filepath_root[-len(self.name):] == self.name:
            checkpoint_filepath_full = checkpoint_filepath_root + '_' + self.name
        else:
            checkpoint_filepath_full = checkpoint_filepath_root
        print('... loading checkpoint from ' + checkpoint_filepath_full + ' ...')
        self.load_state_dict(torch.load(checkpoint_filepath_full))
