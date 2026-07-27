import numpy as np
import pandas as pd
import networkx as nx

# JSON by construction does not recognize numerical keys; only string keys
def json_keys_to_int(json_dict):
    return {int(k): v for k, v in json_dict.items()}

# Get all the triangles from a networkx G 
def all_triangles(G):
    node_to_id = {node: i for i, node in enumerate(G)}
    for u in G:
        u_id = node_to_id[u]
        u_nbrs = G._adj[u].keys()
        for v in u_nbrs:
            v_id = node_to_id.get(v, -1)
            if v_id <= u_id:
                continue
            v_nbrs = G._adj[v].keys()
            for w in v_nbrs & u_nbrs:
                if node_to_id.get(w, -1) > v_id:
                    yield u, v, w
                    
def redux_subgraph(G, lens, D, B=None, prune_leaves=True, verbose=False):
    # The subgraph right after adding D
    dthreshold = lens.loc[D]
    if B is None:
        nodes = lens[lens <= dthreshold].index
    else:
        bthreshold = lens.loc[B]
        nodes = lens[(lens >= bthreshold) & (lens <= dthreshold)].index
        
    subgraph = G.subgraph(nodes)
    if verbose:
        print('Nodes present after adding node [', D, ']:\t', len(subgraph.nodes()), sep='')
    
    # Get the connected component that contains D
    S = [subgraph.subgraph(c).copy() for c in nx.connected_components(subgraph)]
    i= 0
    while D not in S[i].nodes():
        i += 1
    if verbose:
        print('Nodes kept in the connected component:\t', len(S[i].nodes()), sep='')
    
    if prune_leaves:
        # Remove leaves (in the graph theory sense): nodes that are part of a single edge
        uq,ct = np.unique(np.asarray(list(S[i].edges())).ravel(), return_counts=True)
        S[i] = S[i].subgraph(uq[ct>1])
        if verbose:
            print('Nodes after pruning:\t', len(S[i].nodes()), sep='')

    return S[i]
    
def birth_death_path(G, lens, B,D):
    # The subgraph with only the vertices added between timepoints B and D
    b,d = lens.loc[[B,D]]
    subgraph = G.subgraph(lens[(lens <= d) & (lens >=b)].index)
    
    # Return the shortest path between B and D
    return nx.shortest_path(subgraph, B, D)

def is_cyclic(G, source=None, orientation=None):
    is_cycle = True
    try:
        nx.find_cycle(G, source=source, orientation=orientation)
    except nx.NetworkXNoCycle:
        is_cycle = False
    return is_cycle

def generating_cycles(subb, B, neighbors, spath, minlength=4):

    simple_cycles = []
    dneighs = set()
    layer = 0

    while (len(simple_cycles) == 0) & (layer < len(spath)):
        # Get all the neighbors of the nodes in the birth-death shortest path
        # This includes neighbors of D but NOT of B
        # Then consider just the subset of them that have a filtration value lower than B
        # Consider the sub-subgraph made by this subset of vertices
        #print(layer, spath[layer], sep='\t')
        for n in spath[layer]:
            dneighs |= set(neighbors[n])
        dneighs &= set(subb.nodes())
        subg = subb.subgraph(dneighs)

        # Get the simple cycles in this reduced graph
        # Only keep those with B
        for cycle in nx.simple_cycles(subg):
            if (B in cycle) and (len(cycle) > minlength):
                simple_cycles.append(cycle)
        layer += 1
        
    return simple_cycles