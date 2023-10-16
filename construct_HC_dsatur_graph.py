import networkx as nx

def generate_HC_dsatur_graph_as_networkx(n):
    G = nx.Graph()
    no_W_1_nodes = n-2
    no_W_2_nodes = n-1
    no_W_3_nodes = n-1
    no_W_2_extra_vertices = 2*n
    no_W_3_extra_vertices = 2*n
    no_vertices_added = 0
    W_1_vertices = range(no_W_1_nodes)
    no_vertices_added += no_W_1_nodes
    W_2_vertices = range(no_vertices_added, no_vertices_added+no_W_2_nodes)
    no_vertices_added += no_W_2_nodes
    W_3_vertices = range(no_vertices_added, no_vertices_added+no_W_3_nodes)
    no_vertices_added += no_W_3_nodes
    W_2_extra_vertices = range(no_vertices_added, no_vertices_added+no_W_2_extra_vertices)
    no_vertices_added += no_W_2_extra_vertices
    W_3_extra_vertices = range(no_vertices_added, no_vertices_added+no_W_3_extra_vertices)
    
    G.add_nodes_from(W_1_vertices)
    G.add_nodes_from(W_2_vertices)
    G.add_nodes_from(W_3_vertices)
    G.add_nodes_from(W_2_extra_vertices)
    G.add_nodes_from(W_3_extra_vertices)
    
    for W_2_vertex, W_3_vertex in zip(W_2_vertices, W_3_vertices):
        G.add_edge(W_2_vertex, W_3_vertex)
    
    for ind1, W_1_vertex in enumerate(W_1_vertices):
        for ind2, W_2_vertex in enumerate(W_2_vertices):
            if ind1!=ind2:
                G.add_edge(W_1_vertex, W_2_vertex)
                
    for ind1, W_1_vertex in enumerate(W_1_vertices):
        for ind3, W_3_vertex in enumerate(W_3_vertices[ind1:]):
            G.add_edge(W_1_vertex, W_3_vertex)
            
    for W_2_vertex in W_2_vertices:
        vertex_degree = G.degree(W_2_vertex)
        for i in range(2*n-vertex_degree):
            G.add_edge(W_2_vertex, W_2_extra_vertices[i])
            
    for W_3_vertex in W_3_vertices:
        vertex_degree = G.degree(W_3_vertex)
        for i in range(2*n-vertex_degree):
            G.add_edge(W_3_vertex, W_3_extra_vertices[i])
    
    
    return G