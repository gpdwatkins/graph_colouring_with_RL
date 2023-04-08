from gym.envs.registration import register

register(
    id='graph-colouring-v0',
    entry_point='gym_graph_colouring.envs:GraphColouring',
)