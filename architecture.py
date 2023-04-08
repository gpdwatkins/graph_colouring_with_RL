import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, Size
from torch_geometric.data import Data

from principal_neighbourhood_aggregation import AGGREGATORS, SCALERS

DEVICE = "cuda:0"

class MyGN(MessagePassing):
    def __init__(self, message_in_dim, message_out_dim, update_in_dim, update_out_dim, aggregators, scalers, avg_deg):

        super(MyGN, self).__init__(aggr=None)

        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]
        self.avg_deg=avg_deg

        msg_mlp_hidden_dim = 64
        upd_mlp_hidden_dim = 64
        
        self.message_mlp = nn.Sequential(
            nn.Linear(message_in_dim, msg_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(msg_mlp_hidden_dim, message_out_dim)
            )

        if not (update_in_dim is None or update_out_dim is None):
            self.update_mlp = nn.Sequential(
                nn.Linear(update_in_dim, upd_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(upd_mlp_hidden_dim, update_out_dim)
                )

        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def message(self, x_j, x_i, edge_attr):
        # Message function takes in:
        # - vertex features of the source vertex x_j
        # - vertex features of the dest vertex x_i
        # - edge features of e_ji
        # and outputs:
        # - encoding of e_ji
        message_nn_input = torch.cat([x_j, x_i, edge_attr], dim=1)
        message_nn_output = self.message_mlp(message_nn_input)
        return message_nn_output

    def aggregate(self, inputs, index, edge_attr, dim_size=None):
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)
        return out

    def update(self, aggr_out, x):
        # Update function takes in:
        # - Aggregation of edges arriving at vertex i
        # - vertex features of vertex x_i
        # and outputs:
        # - vertex encoding of x_i
        if type(x) is tuple:
            x=x[1]
        update_nn_input = torch.cat([aggr_out, x], dim=1)
        update_nn_output = self.update_mlp(update_nn_input)
        return update_nn_output

    def propagate(self, edge_index: Adj, size: Size = None, do_update_step=True, **kwargs):
        """The initial call to start propagating messages."""
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
    def __init__(self, vertex_features_size, edge_features_size, global_features_size, output_size, name, avg_degrees = None, alpha=None):
        """Initialize parameters and build model.
        Params
        ======
            observation_size (int): Dimension of each observation
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(GNNetwork, self).__init__()

        self.name = name

        msg_output_size = 64
        upd_output_size = 64

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity']

        gn1_msg_input_size = 2*vertex_features_size + edge_features_size
        gn1_msg_output_size = msg_output_size
        gn1_upd_input_size = len(aggregators) * len(scalers) * gn1_msg_output_size + vertex_features_size
        gn1_upd_output_size = upd_output_size
        
        gn2_msg_input_size = 2*gn1_upd_output_size + gn1_msg_output_size
        gn2_msg_output_size = msg_output_size
        gn2_upd_input_size = len(aggregators) * len(scalers) * gn2_msg_output_size + gn1_upd_output_size
        gn2_upd_output_size = upd_output_size

        gn3_msg_input_size = 2*gn2_upd_output_size + gn2_msg_output_size
        gn3_msg_output_size = msg_output_size
        gn3_upd_input_size = len(aggregators) * len(scalers) * gn3_msg_output_size + gn2_upd_output_size
        gn3_upd_output_size = upd_output_size

        gn4_msg_input_size = 2*gn3_upd_output_size + gn3_msg_output_size
        gn4_msg_output_size = msg_output_size
        gn4_upd_input_size = len(aggregators) * len(scalers) * gn4_msg_output_size + gn3_upd_output_size
        gn4_upd_output_size = upd_output_size

        gn5_msg_input_size = 2*gn4_upd_output_size + gn4_msg_output_size
        gn5_msg_output_size = msg_output_size
        gn5_upd_input_size = len(aggregators) * len(scalers) * gn5_msg_output_size + gn4_upd_output_size
        gn5_upd_output_size = upd_output_size

        fc1_units=64
        fc2_units=64

        # In first GN layer, the input to update is a concatenation of:
        # - Aggregated message output
        # - original vertex feature
        # In second GN layer, the input to update is a concatenation of:
        # - Aggregated message output
        # - vertex embedding from first GN layer
        
        self.gn_layer1 = MyGN(gn1_msg_input_size, gn1_msg_output_size, gn1_upd_input_size, gn1_upd_output_size, aggregators, scalers, avg_deg=avg_degrees)
        self.gn_layer2 = MyGN(gn2_msg_input_size, gn2_msg_output_size, gn2_upd_input_size, gn2_upd_output_size, aggregators, scalers, avg_deg=avg_degrees)
        self.gn_layer3 = MyGN(gn3_msg_input_size, gn3_msg_output_size, gn3_upd_input_size, gn3_upd_output_size, aggregators, scalers, avg_deg=avg_degrees)
        self.gn_layer4 = MyGN(gn4_msg_input_size, gn4_msg_output_size, gn4_upd_input_size, gn4_upd_output_size, aggregators, scalers, avg_deg=avg_degrees)
        self.gn_layer5 = MyGN(gn5_msg_input_size, gn5_msg_output_size, gn5_upd_input_size, gn5_upd_output_size, aggregators, scalers, avg_deg=avg_degrees)

        self.fc1 = nn.Linear(gn5_upd_output_size+global_features_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)

        self.dummy_data = Data(x=torch.tensor([]), edge_index = torch.tensor([], dtype=torch.long), edge_attr = torch.tensor([]))

        if not alpha is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, data, global_features, vertex_batch_map, edge_batch_map, called_by=None):
        self.dummy_data.edge_index = data.edge_index
        self.dummy_data.edge_attr, self.dummy_data.x = self.gn_layer1(data)
        self.dummy_data.edge_attr, self.dummy_data.x = self.gn_layer2(self.dummy_data)
        self.dummy_data.edge_attr, self.dummy_data.x = self.gn_layer3(self.dummy_data)
        self.dummy_data.edge_attr, self.dummy_data.x = self.gn_layer4(self.dummy_data)

        _, vertex_embeddings = self.gn_layer5(self.dummy_data)

        global_features_repeated_for_vertices = global_features[vertex_batch_map]
        y = torch.cat((vertex_embeddings, global_features_repeated_for_vertices), dim=1)

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return vertex_embeddings, y

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
