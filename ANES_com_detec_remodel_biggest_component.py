import matplotlib
matplotlib.use('agg')

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import igraph as ig
import time
import sys
#import graph_tool.all as gt
import pyintergraph


def find_communities(G, number_components=2):
    """
    Insert graph, choose the biggest component og the graph, then compute the edges which have to be removed to split it
    up into two communities. It works like the Girvan-Newman algorithm: hierarchical method to detect commmunities in
    complex systems. Compute with the igraph (performance-reasons) the edge with the highest betweenness and erase it.
    Check whether the graph split up into two. If not, repeat.
    :param G: networkx-graph
    :param number_components: number of components [Stop-argument]
    :return: H = the biggest connected component as a networkx-graph;
            sub_H = dict of connected components of H for different amout of components
             edges_between_comp = list of tuples of erased edges;
             mapping = list of the nodes in each component
    """
    # getting the biggest component
    if number_components <=2:
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G_subgraph = G.subgraph(Gcc[0]).copy()
        H = nx.Graph()
        H.add_nodes_from(G_subgraph.nodes)
        for (u,v,d) in G_subgraph.edges(data=True):
            H.add_edges_from([(u,v)])
            H[u][v]['weight'] = d['weight']
      #  H.add_edges_from(G_subgraph.edges)
    else:
        H = G

    mapping = dict()
    i = 0
    for node_name in H.nodes:
        mapping[node_name] = i
        i += 1
    H = nx.relabel_nodes(H, mapping)
    # print(H.nodes.values())

    # Girvan-Newman algorithm
    num_comp = nx.number_connected_components(H)
    # print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
    count = 0
    #print(f'Num_comp is = {num_comp}')
    edges_between_comp = []
    if not number_components == 1:
        while num_comp < number_components:
            u, v = edge_highest_betweenness(H) #edge_highest_betweenness(H) #edge_highest_betweenness_with_edgeweight(H)
            H.remove_edge(u, v)
            edges_between_comp.append((u, v))
            num_comp = nx.number_connected_components(H)
            sys.stdout.write('\r' + "." * (1 + (count % 3)))
            count += 1
        sys.stdout.write('\r' + "finished GN \n")

    # changing back the names:
    mapping = {y: x for x, y in mapping.items()}
    H = nx.relabel_nodes(H, mapping)
    # print(H.nodes.values())
    # define add nodes to component
    sub_H = [c for c in sorted(nx.connected_components(H), key=len, reverse=True)]
    
    # defining components
    for node in H.nodes():
        for i in range(len(sub_H)):
            if node in sub_H[i]:
                H.nodes[node]['component'] = i
    # all nodes of graph or only component?
    only_component = True
    G = H if only_component else G
    return H, sub_H, edges_between_comp, mapping


def select_data_from_csv(csv, var_names, read_until=None, sep=',', anes2016=None):
    """
    Select data from csv.-file and return a numpy-vstack
    :param csv: string
    :param vars: list of strings; used variables
    :param read_until: int; default=None -> read max data length
    :param sep: string; default=','; seperator of .csv-file
    :return: list
    """
    # ['V160001', 'V161155', 'V161232', 'V161189', 'V161192', 'V161209', 'V161231', 'V161201', 'V161187']
    complete_data = pd.read_csv(csv, sep=sep, low_memory=False)
    selected_data = complete_data[var_names].copy()
    short_selected_data = selected_data.iloc[:read_until, [0, 1, 2, 3, 4, 5, 6, 7, 8]]  # without patriot flag
    data_list = short_selected_data.values.tolist()

    print(f'1. Select Lines {len(data_list)} with data')

    # remove negative values from data list:  after length = 4348 values
    lines_to_remove = []
    for line in data_list:
        line_attributes = line[2:9]
        check = False
        for i in line_attributes:
            if (i == 99 or i < 0) and not check:
                check = True
        # remove 5 in abortion first feature in list 5=other; rescaled ones!!!
        if anes2016 and line[2] == 5:
            check = True
        if check:
            lines_to_remove.append(line)

    unvalid = len(lines_to_remove)

    # remove all collected negative lines from data
    while lines_to_remove:
        data_list.remove(lines_to_remove.pop())

    print(f'2. {unvalid} lines erased. Number of valid rows in data: {len(data_list)}')

    return data_list #np.vstack(data_list)  # return MATRIX


def edge_highest_betweenness(G):
    """
    Calculate highest edge betweenness with igraph
    :param G: graph
    :return: edge with highest edge betweenness in G
    """
    G_ig = ig.Graph(len(G.nodes.values()))
    G_ig.add_edges(G.edges)
    edges_betw_list = G_ig.edge_betweenness()
    gig_edge_list = G_ig.get_edgelist()

    return gig_edge_list[edges_betw_list.index(max(edges_betw_list))]


def edge_highest_betweenness_graph_tool(G):
    """
    Calculate highest edge betweenness with igraph
    :param G: graph
    :return: edge with highest edge betweenness in G
    """
    #build graph
    g_gt = gt.Graph(directed=False)
    H = nx.convert_node_labels_to_integers(G)
    g_gt.add_edge_list(H.edges())

    #get edge_betweenness
    node_betweenness, edge_betweenness = gt.betweenness(g_gt)

    # list of edges
    edges_gt = edge_betweenness.get_graph().get_edges()

    # return edge with highest edge betweenness - edge_betweenness.a = Propertymap with edgebetweeness
    return edges_gt[list(edge_betweenness.a).index(edge_betweenness.a.max())]


def community_detection_lead_eigen(G, n_clusters):
    H = nx.convert_node_labels_to_integers(G)
    G_ig = ig.Graph(len(H.nodes.values()))
    G_ig.add_edges(H.edges)
    #                                                               ADDING 10 to weight!!!!!
    G_ig.es['weight'] = [10 + weight['weight'] for (u,v,weight) in H.edges(data=True)] ### THAT IS A GOOD QUESTION?

    # infomap as community detection algorithm
    communities =G_ig.community_leading_eigenvector(n_clusters, weights='weight')
    # give out commmunity for each node as list
    # no need to change it back

    return communities.membership


def edge_highest_betweenness_with_edgeweight(G):
    """
    Calculate highest edge betweenness with igraph with edge weight

    TO USE it with the edge definition of EDGE WEIGHT: HIgh EDGEWEIGHT means CLOSER together. The EDGE BETWEENNESS
    calculates the costs. --> We devide ::> 1/weight -- Answer for now: NO, we do not do that

    :param G: graph
    :return: edge with highest edge betweenness in G
    """
    G_ig = ig.Graph(len(G.nodes.values()))
    G_ig.add_edges(G.edges)
    #                                                               ADDING 10 to weight!!!!!
    G_ig.es['weight'] = [weight['weight'] for (u,v,weight) in G.edges(data=True)] ### THAT IS A GOOD QUESTION?

    edges_betw_list = G_ig.edge_betweenness(weights='weight')
    gig_edge_list = G_ig.get_edgelist()
    return gig_edge_list[edges_betw_list.index(max(edges_betw_list))]


def build_threshold_graph(data_list, threshold, rescaled_data_list, graph_size=0):
    """
    build graph G with data list. Edges connecting similar nodes, whose similarity is above a threshold. Over more,
    rescales the attitudes to a range between -1 and 1.
    :param data_list: list; attitudes of agents
    :param threshold: int; minimal feature overlap for edge
    :param attribute_list: list of strings; names of attitudes
    :param graph_size: int; default=0 -> maximal graph size; count of nodes in the graph
    :param new_scale: dict; rescaling values for each attitude
    :return: graph
    """
    # construct graph
    # select i.e. the first 100 valid nodes from data
    select_data_size = graph_size if graph_size else len(data_list)
    # build graph with nodes
    G = nx.Graph()
    names = [data_list[i][0] for i in range(select_data_size)]
    G.add_nodes_from(names)

    rescaled_features = dict()
    # feed axelrod with data
    for row, name in enumerate(G.nodes):
        rescaled_features[name] = rescaled_data_list[row] #data_list[row][2:]

    # min of equalty
    # adjust weight for each node pair: -7 to 7
    edge_counter = 0
    for node in G.nodes:
        for node_neighbour in G.nodes:
            if node != node_neighbour and not G.has_edge(node_neighbour, node):
                # weight = 8 - (0-16) # 8 same, -8 totally different
                weight = len(rescaled_data_list[0]) - sum(abs(np.array(rescaled_features[node])
                                                       - np.array(rescaled_features[node_neighbour])))
                if weight >= threshold:
                    edge_counter +=1
                    G.add_edge(node, node_neighbour)
                    G[node][node_neighbour]['weight'] = weight

    print(f' The graph is build up with a threshold of {threshold}')
    return G, rescaled_features


def build_graph_auto_threshold(data_list, rescaled_data_list, graph_size=0):
    """
    build graph G with data list. Edges connecting similar nodes, whose similarity is above a threshold. Over more,
    rescales the attitudes to a range between -1 and 1.
    :param data_list: list; attitudes of agents
    :param threshold: int; minimal feature overlap for edge
    :param attribute_list: list of strings; names of attitudes
    :param graph_size: int; default=0 -> maximal graph size; count of nodes in the graph
    :param new_scale: dict; rescaling values for each attitude
    :return: graph
    """
    # construct graph
    # select i.e. the first 100 valid nodes from data
    select_data_size = graph_size if graph_size else len(data_list)
    # build graph with nodes
    G = nx.Graph()
    names = [data_list[i][0] for i in range(select_data_size)]
    G.add_nodes_from(names)

    rescaled_features = dict()
    # feed axelrod with data
    for row, name in enumerate(G.nodes):
        rescaled_features[name] = rescaled_data_list[row] #data_list[row][2:]

    # min of equalty
    # adjust weight for each node pair: -7 to 7

    # set threshold on maximum
    threshold_step = 0.5
    threshold = len(rescaled_data_list[0]) - threshold_step
    best_threshold_found = False
    while not best_threshold_found:
        for node in G.nodes:
            for node_neighbour in G.nodes:
                if node != node_neighbour and not G.has_edge(node_neighbour, node):
                    # weight = 8 - (0-16) # 8 same, -8 totally different
                    weight = len(rescaled_data_list[0]) - sum(abs(np.array(rescaled_features[node])
                                                           - np.array(rescaled_features[node_neighbour])))

                    # ===========================================================================weight between 0 and 1
                    #double_number_of_f = 2*len(rescaled_data_list[0])
                    #weight = sum(abs(np.array(rescaled_features[node]) - np.array(rescaled_features[node_neighbour])))/double_number_of_f

                    if weight >= threshold:
                        G.add_edge(node, node_neighbour, weight=weight)
                        G[node][node_neighbour]['weight'] = weight
        H_graph, biggest_sub_graph, *args = find_communities(G, 1)
        if len(biggest_sub_graph[0]) < 0.8 * G.number_of_nodes():
            print(f'{threshold:.2f}')
            threshold -= threshold_step
        else:
            best_threshold_found = True

    print(f' The graph is build up with a threshold of {threshold:.1f}')
    return G, rescaled_features, threshold


def draw_community_graph_without_splitup(G, H, sub_H, data_list, edges_between_comp, mapping):
    """
    Draw and save graph as png-file and as gephi-graph
    :param G: graph
    :param H: biggest connected component in graph
    :param sub_H: communities in graph
    :return: None
    """
    component = {}
    # all nodes of graph or only component?
    only_component = True
    G = H if only_component else G
    for node in G.nodes:
        for component_number, component_list in enumerate(sub_H):
            if node in component_list:
                component[node] = component_number
                break
    # print(component)
    nx.set_node_attributes(G, component, 'component')

    pos = nx.random_layout(G)

    comp_pos = {}
    for x, y in pos.items():
        if G.nodes[x]['component'] == 1:
            comp_pos[x] = [y[0] + 1, y[1] + 1]
        else:
            comp_pos[x] = [y[0], y[1]]

    # names of nodes and party-values
    keys = [data_list[i][0] for i in range(len(data_list))]
    values = [data_list[i][1] for i in range(len(data_list))]
    party_affiliation_dict = dict(zip(keys, values))

    democrats = [];
    republicans = [];
    neither = []
    party = dict()
    for node in G.nodes:  # simply a list
        if node in H.nodes:
            if party_affiliation_dict[node] == 1:
                party[node] = 'democrat'
                democrats.append(node)
            elif party_affiliation_dict[node] == 2:
                party[node] = 'republican'
                republicans.append(node)
            else:
                party[node] = 'unknown'
                neither.append(node)
    nx.set_node_attributes(G, party, 'party')

    # print("subH_length===", len(sub_H))
    # print component
    parties_in_component_dict = dict()
    for components in range(len(sub_H)):
        parties_in_component_dict[components] = {'republican': 0, 'democrat': 0, 'unknown': 0}

    for node in G.nodes.values():
        parties_in_component_dict[node['component']][node['party']] += 1

    print(parties_in_component_dict)

    for component_num in range(len(sub_H)):
        print(f'In component {component_num} with size: {len(sub_H[component_num])}'
              f'  are: {parties_in_component_dict[component_num]}')

    # visualize
    plt.style.use('dark_background')

    nx.draw_networkx_nodes(G, comp_pos, node_list=neither, node_color='tab:orange', alpha=0.9, node_size=50)
    nx.draw_networkx_nodes(G, comp_pos, nodelist=democrats, node_color='tab:blue', alpha=0.9, node_size=50)
    nx.draw_networkx_nodes(G, comp_pos, nodelist=republicans, node_color='tab:red', alpha=0.9, node_size=50)

    edges_between_comp_renamed = []
    for edge in edges_between_comp:
        edges_between_comp_renamed.append((mapping[edge[0]], mapping[edge[1]]))

    nx.draw_networkx_edges(G, comp_pos, alpha=0.3, style='--', edge_color='white')
    nx.draw_networkx_edges(G, comp_pos, alpha=0.5, style='--', edge_color='white', edgelist=edges_between_comp_renamed)
    # 'fuchsia'
    # save_data
    nx.write_gexf(G, 'ALL_2016_ANES_graph.gexf')
    if len(sub_H) == 2:
        plt.savefig("graph_two_comp.png")
    plt.close()
    plt.style.use('default')
    return comp_pos


def save_community_graph(G, H, sub_H, data_list, edges_between_comp, mapping):
    """
    Draw and save graph as png-file and as gephi-graph
    :param G: graph
    :param H: biggest connected component in graph
    :param sub_H: communities in graph
    :return: None
    """
    component = {}
    # all nodes of graph or only component?
    only_component = True
    G = H if only_component else G
    for node in G.nodes:
        for component_number, component_list in enumerate(sub_H):
            if node in component_list:
                component[node] = component_number
                break
    #print(component)
    nx.set_node_attributes(G, component, 'component')

    pos = nx.spring_layout(G)

    comp_pos = {}
    for x, y in pos.items():
        if G.nodes[x]['component'] == 1:
            comp_pos[x] = [y[0], y[1]]  #[y[0] + 0.5, y[1] + 0.5]
        else:
            comp_pos[x] = [y[0], y[1]]

    # names of nodes and party-values
    keys = [data_list[i][0] for i in range(len(data_list))]
    values = [data_list[i][1] for i in range(len(data_list))]
    party_affiliation_dict = dict(zip(keys, values))

    democrats = []; republicans = []; neither = []
    party = dict()
    for node in G.nodes:  # simply a list
        if node in H.nodes:
            if party_affiliation_dict[node] == 1:
                party[node] = 'democrat'
                democrats.append(node)
            elif party_affiliation_dict[node] == 2:
                party[node] = 'republican'
                republicans.append(node)
            else:
                party[node] = 'unknown'
                neither.append(node)
    nx.set_node_attributes(G, party, 'party')

    #print("subH_length===", len(sub_H))
    # print component
    parties_in_component_dict = dict()
    for components in range(len(sub_H)):
        parties_in_component_dict[components] = {'republican': 0, 'democrat': 0, 'unknown': 0}

    for node in G.nodes.values():
        parties_in_component_dict[node['component']][node['party']] += 1

    print(parties_in_component_dict)

    for component_num in range(len(sub_H)):
        print(f'In component {component_num} with size: {len(sub_H[component_num])}'
              f'  are: {parties_in_component_dict[component_num]}')

    # visualize
    plt.style.use('dark_background')
    
    nx.draw_networkx_nodes(G, comp_pos, nodelist=neither, node_color='tab:orange', alpha=0.9, node_size=50)
    nx.draw_networkx_nodes(G, comp_pos, nodelist=democrats, node_color='tab:blue', alpha=0.9, node_size=50)
    nx.draw_networkx_nodes(G, comp_pos, nodelist=republicans, node_color='tab:red', alpha=0.9, node_size=50)

    edges_between_comp_renamed = []
    for edge in edges_between_comp:
        edges_between_comp_renamed.append((mapping[edge[0]], mapping[edge[1]]))

    nx.draw_networkx_edges(G, comp_pos, alpha=0.3, style='-', edge_color='white')
    nx.draw_networkx_edges(G, comp_pos, alpha=0.25, style='--', edge_color='fuchsia', edgelist=edges_between_comp_renamed)
                                                                        # 'fuchsia'
    # save_data
    ######## delete for visualize#########nx.write_gexf(G, 'ALL_2016_ANES_graph.gexf')
    ######## delete for visualize#########if len(sub_H) == 2:
    ######## delete for visualize#########    plt.savefig("graph_two_comp.png")
    ######## delete for visualize#########plt.close()
    ######## delete for visualize#########plt.style.use('default')
    ##return comp_pos
    return edges_between_comp_renamed


def get_wss(G, rescaled_features, number_cluster):
        # mapping has all relevant information
    # build and split the Q_matrix into the clusters
    nodes = dict()
    for node, component in G.nodes(data='component'):
        if not nodes.get(component):
            nodes[component] = []
        nodes[component].append(node)

    # Q_component
    Q_comp = dict()
    for component in range(number_cluster):
        for node in nodes[component]:
            if not Q_comp.get(component):
                Q_comp[component] = []
            Q_comp[component].append([node, *rescaled_features[node]])
        Q_comp[component] = np.vstack(Q_comp[component])

    wss = 0
    for Q_comp_submatrix in Q_comp.values():
        for feature in range(1, 8):
            mean = Q_comp_submatrix[:, feature].mean()
            for el in Q_comp_submatrix[:, feature]:
                wss += (el-mean)**2
    return wss


# if __name__ == '__main__':
def get_graph(data_list, rescaled_data_list, max_number_cluster, minimum_size_for_two_comp=False, threshold=None):
    """

    :param data_list: data_list which have stored the selected variables
    :param max_number_cluster: int; when to stop splitting up the biggest component
    :param minimum_size_for_two_comp: defines if there is a resizing if components are too small! (only for comp = 2)
    :return: dictionary of graphs; rescaled data
    """
    # Main CODE
    # Graph with nodes and edges over threshold

    if threshold:
        G, rescaled_features = build_threshold_graph(data_list, threshold, rescaled_data_list)
    else:
        G, rescaled_features, threshold = build_graph_auto_threshold(data_list, rescaled_data_list=rescaled_data_list)


    # find two biggest communities in graph
    # getting the biggest component, search until it the two components have a minimal size

    G_dict = dict()
    edges_between_comp_renamed = None
    for number_cluster in range(1, max_number_cluster+1):
        H, sub_H, edges_between_comp, mapping = find_communities(G, number_cluster)

        # to print
        print('Biggest component:')
        print('               number of nodes: ', H.number_of_nodes())
        print('               number of edges: ', H.number_of_edges())
        number_of_edges = H.number_of_edges()
        count = len(edges_between_comp)
        # to break it into 2 components which are are "similar" in size
        if number_cluster == 2:
            resplit = 0
            if minimum_size_for_two_comp:
                while len(sub_H[0]) * 0.1 > len(sub_H[1]):
                    H, sub_H, edges_between_comp, mapping = find_communities(H, number_cluster)

                    resplit +=1
                    count += len(edges_between_comp)
            print(f"{resplit} times re-split up [2. cluster was too small]")
            print('[1/2] component: ', sub_H[0])
            print('[2/2] component: ', sub_H[1])
        to_print_comp_length = []
        for sub_H_graph in sub_H:
            to_print_comp_length.append(len(sub_H_graph))
        print(f'erased {count} edges from the graph and Num_comp is now {len(sub_H)}: {to_print_comp_length}')
        # to print only graph with biggest component
        G = H
        print(edges_between_comp, number_of_edges)
        
        if number_cluster == 2:  # normally should be 2
            edges_between_comp_renamed = save_community_graph(G, H, sub_H, data_list, edges_between_comp, mapping)
        G_dict[number_cluster] = G

    #### for the WELLCOME TRUST DATA I NEED ALSO THE COUNT AND THRESHOLD
    #return G_dict, rescaled_features, threshold, count, number_of_edges
    


    finish_time = time.time()
    #print('total time: %.2f s' % (finish_time - start_time))
    return G_dict, rescaled_features, threshold, edges_between_comp_renamed

#nx.write_gexf(G, 'ALL_2016_ANES_graph.gexf')

#plt.savefig("test.png")

#plt.show()

"""
#####################rescaling also could be done like:

"""
"""
new_scale_dict = {'abortpre_4point': [- 1, -(1/3), (1/3), 1],  # new scale 1-4
                  'guarpr_self': [-1, -(2/3), -(1/3), 0, (1/3), (2/3), 1],  # stay the same
                  'immig_policy': [-1, -(1/3), (1/3), 1],  # stay the same
                  'fedspend_welfare': [1, -1, 0],  # 1=positive 2=negative 3=neutral
                  'gayrt_marry': [1, 0, -1],  # changed 1=postive, 3=negative
                  'envjob_self': [-1, -(2/3), -(1/3), 0, (1/3), (2/3), 1],  # No regulation by gov:
                  'gun_control': [1, -1, 0]  # , # 1=postive, 2=negative, 3=neutral
                  # 'patriot_flag': [1, 0.5, 0, -0.5, -1]# changed 5, 4, 3, 2, 1
                  }
# old scale
new_scale_dict = {'abortpre_4point': [- 1, -0.5, 0, 0.5, 1],  # scale 1-5
                  'guarpr_self': [-1, -(2/3), -(1/3), 0, (1/3), (2/3), 1],  # stay the same
                  'immig_policy': [-1, -(1/3), (1/3), 1],  # stay the same
                  'fedspend_welfare': [-1, 0, 1],  # 1=positive 2=negative 3=neutral
                  'gayrt_marry': [-1, 0, 1],  # changed 1=postive, 3=negative
                  'envjob_self': [-1, -(2/3), -(1/3), 0, (1/3), (2/3), 1],  # No regulation by gov:
                  'gun_control': [-1, 0, 1]  # , # 1=postive, 2=negative, 3=neutral
                  # 'patriot_flag': [1, 0.5, 0, -0.5, -1]# changed 5, 4, 3, 2, 1
                  }
"""
