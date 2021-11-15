#  Hierarchical clustering by A. Dinkelberg
import csv
import itertools
from copy import deepcopy
import matplotlib
matplotlib.use('agg')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
from ANES_com_detec_remodel_biggest_component import get_graph
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
import random
#import seaborn as sns
import sys
from multiprocessing import Process, Queue
#from kneed import KneeLocator
from scipy.stats import norm

# supress ClusterWarnings
import warnings
warnings.filterwarnings("ignore")


def elbow_plot_for_other_community_dection(d_matrix, rescaled_data_list, max_amount_clusters, predicted_values, plot_name=""):
    """
    Calculate the within square of sum for 1 to max amount of clusters. Generating elbow plot.
    For Hierarchical clustering OR Girvan-Newman method. Without KMeans, using for-loops to calculate
    method.
    Saves the plot.
    :param d_matrix: distance matrix
    :param reduce_data_hc: list;  names of nodes which are in the giant component
    :param max_amount_clusters: max number of clusters
    :param predicted_values: dictionary of community assignments
    :return: dictionary of with clusters
    """

    cluster_count = range(1, max_amount_clusters + 1)
    clusters_dict = dict()  # to save results of node in clusters
    wss = []
    # if d_matrix is dictionary --> calculating wss for GN algorithm

    label = plot_name

    for size in cluster_count:
        wss_complete, clusters_dict[size] = get_wss_HC(rescaled_data_list, d_matrix, size, component_list=predicted_values[size])
        wss.append(wss_complete)

    ### >>>>>>>plotting
    print(cluster_count, wss)
    elbow = KneeLocator(cluster_count, wss, curve="convex", direction="decreasing").knee
    if plot_name:
        plt.close()
        print(elbow)
        
        plt.plot(cluster_count, wss, 'o', label=label, linestyle='--')
        if elbow:
            plt.plot(cluster_count[elbow - 1], wss[elbow - 1], 'bx')
        plt.title('double elbow')
        plt.xlabel('Number of clusters')
        plt.ylabel('WSS')
        plt.legend()
        plt.savefig(f'WSS_test_{plot_name}')

    return wss



def create_random_dataset_different_mus(n_agents=100, n_questions=10, scale_steps=10, mu_max=5, number_ranks=5, n_comp=2,
                                        sd=0.5, split_up=[0.5, 0.5]):

    ranks = int((n_questions / number_ranks)+0.5)

    number_of_questions = [n_questions]
    while sum(number_of_questions) > n_questions-1:
        number_of_questions = [int(np.random.normal(ranks, scale=1.5)) for _ in range(number_ranks-1)]
        for el in number_of_questions:
            if el < 1:
                number_of_questions = [10]
    # plus rest?
    mu_dist_multi = []

    step_size = 0.2
    if mu_max > scale_steps - step_size:
        raise ValueError(f'max mu is to high [{mu_max}]: it has to be lower than {scale_steps - step_size}')

    mus = [i for i in np.arange(1, mu_max, step_size)]

    for i in range(number_ranks-1):
        if not mus:
            mus.append(mu_max)
        choice = random.choice(mus)
        mu_dist_multi.append(choice)
        mus.remove(choice)


    mu_dist_multi.sort(reverse=True)

    data_set = pd.DataFrame()
    item_name = 1

    names = []

    for q_per_mu, mu_dist in zip(number_of_questions, mu_dist_multi):
        data_columns = create_sim_dataset_ranking(n_agents, q_per_mu, scale_steps, mu_dist=mu_dist, n_comp=n_comp,
                                                  higher_sd_q=0, random_q=0, sd=sd, split_up=split_up)

        end = item_name+q_per_mu
        for name_id in range(item_name, end):
            name = "Q_" + str(name_id)
            names.append(name)
            data_set[name] = data_columns[list(data_columns.columns)[name_id-end]]

        item_name += q_per_mu

    random_questions = n_questions - sum(number_of_questions)

    random_data = create_sim_dataset_ranking(n_agents, random_questions, scale_steps, mu_dist=mu_dist, n_comp=n_comp,
                                             higher_sd_q=0, random_q=random_questions, sd=sd, split_up=split_up)

    for name_id in range(sum(number_of_questions)+1, n_questions+1):
        name = "Q_" + str(name_id)
        names.append(name)
        data_set[name] = random_data[list(random_data.columns)[name_id - n_questions -1]]

    data_set.insert(0, 'party_aff', random_data['party_aff'])
    data_set.insert(0, 'person_id', random_data['person_id'])

    number_of_questions.append(random_questions)
    ranking = {}
    name_pos = 0
    for i, q in enumerate(number_of_questions):
        for _ in range(q):
            ranking[names[name_pos]] = (i+1)
            name_pos += 1
    return data_set, ranking


def create_sim_dataset_ranking(n_agents=100, n_questions=10, scale_steps=10, higher_sd_q=0, random_q=0, n_comp=2, mu_dist=1, sd=0.5, split_up=[0.5, 0.5]):
    print('SPLITUP AND COMPNUMBER:', split_up, n_comp)
    if not (higher_sd_q + random_q <= n_questions):
        raise ValueError(f"higher_sd_q and random_q must be together lower than {n_questions}")

    if not len(split_up) == n_comp or not (sum(split_up) == 1):
        raise ValueError(f"length of split up must be ={n_comp} and have a sum of 1!")

        # column names
    column_names = ['person_id', 'party_aff']
    for i in range(n_questions):
        column_names.append('Q_' + str(i + 1))

    # mu to start is calculated
    mu_basis = mu_dist

    mu = (scale_steps + 1) / 2 - mu_dist * int(n_comp / 2) + (n_comp % 2 == 0) * mu_dist / 2

    # answers for each group are computed; every individual from a group answers
    # a set of questions relatively equal (normaldistribution around a mu)
    options_most_relevant = {}
    options_normal={}
    for option in range(1, n_comp + 1):
        if mu < 1 or mu > scale_steps:
            raise ValueError(f'mu_dist={mu_dist} is badly chosen')
        # int cuts from 1.8 to 1; so 0.5 add to get round correct.
        temp = list(map(int, np.random.normal(mu, sd, 100000) + 0.5))
        temp2 = list(map(int, np.random.normal(mu, sd+1.5, 100000) + 0.5))
        options_most_relevant[option] = [in_scale for in_scale in temp if in_scale > 0 and in_scale < (scale_steps + 1)]
        options_normal[option] = [in_scale for in_scale in temp2 if in_scale > 0 and in_scale < (scale_steps + 1)]
        # print(options[option])
        mu += mu_basis

    data_dict = {}
    comp = 1
    change_questions = int(n_questions / n_comp)
    arrow = 2

    # setup for each individual:
    divide_individuals = split_up[comp - 1] * n_agents
    for individual in range(1, n_agents + 1):
        # all answers are unified distributed; individual answers all
        data_dict[individual] = [individual, comp] + [random.randint(1, scale_steps) for i in range(n_questions)]

        # add most important questions
        max_q = len(data_dict[individual])
        for item in range(arrow, max_q - random_q - higher_sd_q):
            data_dict[individual][item] = random.choice(options_most_relevant[comp])
        # add less important questions
        for item in range(max_q - random_q - higher_sd_q, max_q - random_q):
            data_dict[individual][item] = random.choice(options_normal[comp])

        # print(items_to_change, data_dict[individual])

        # division of the groups, increases component
        if int(divide_individuals) == individual:
            comp += 1
            if comp - 1 < len(split_up):
                divide_individuals += split_up[comp - 1] * n_agents

    # makes a DataFrame out of the dictionary and ads ID
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=column_names)
    df.index.name = 'person_id'

    return df


def create_simulated_dataset_with_random_q(n_agents=100, n_questions=10, scale_steps=10, random_q=0, n_comp=2, mu_dist=1, sd=0.5, split_up=[0.5, 0.5]):

    if not len(split_up) == n_comp or not (sum(split_up) == 1):
        raise ValueError(f"length of split up must be ={n_comp} and have a sum of 1!")

        # column names
    column_names = ['person_id', 'party_aff']
    for i in range(n_questions):
        column_names.append('Q_' + str(i + 1))

    # mu to start is calculated
    mu_basis = mu_dist

    mu = (scale_steps + 1) / 2 - mu_dist * int(n_comp / 2) + (n_comp % 2 == 0) * mu_dist / 2

    # answers for each group are computed; every individual from a group answers
    # a set of questions relatively equal (normaldistribution around a mu)
    options = {}
    for option in range(1, n_comp + 1):
        if mu < 1 or mu > scale_steps:
            raise ValueError(f'mu_dist={mu_dist} is badly chosen')
        # int cuts from 1.8 to 1; so 0.5 add to get round correct.
        temp = list(map(int, np.random.normal(mu, sd, 100000) + 0.5))
        options[option] = [in_scale for in_scale in temp if in_scale > 0 and in_scale < (scale_steps + 1)]
        # print(options[option])
        mu += mu_basis

    data_dict = {}
    comp = 1
    change_questions = int(n_questions / n_comp)
    arrow = 2

    # setup for each individual:
    divide_individuals = split_up[comp - 1] * n_agents
    for individual in range(1, n_agents + 1):
        # all answers are unified distributed; individual answers all
        data_dict[individual] = [individual, comp] + [random.randint(1, scale_steps) for i in range(n_questions)]

        for item in range(arrow, len(data_dict[individual]) - random_q):
            data_dict[individual][item] = random.choice(options[comp])

        # print(items_to_change, data_dict[individual])

        # division of the groups, increases component
        if int(divide_individuals) == individual:
            comp += 1
            if comp - 1 < len(split_up):
                divide_individuals += split_up[comp - 1] * n_agents

    # makes a DataFrame out of the dictionary and ads ID
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=column_names)
    df.index.name = 'person_id'

    return df


def create_simulated_dataset_fixed_mu(n_agents=100, n_questions=10, scale_steps=10, n_comp=2, mu_dist=1, sd=0.5, split_up=[0.5, 0.5]):
    """
    
    Creation of answers for a simulated data set. Creates a Dataframe of
    n_agents which answered n_questions of a scale from 1 to scale_steps.
    The data is devided into to n_comp components due to the fact that the
    individuals answers around (1/n_comp) equally, regarding to their group.
    
    Waring: n_comp <= n_questions must hold! 

    Parameters
    ----------
    n_agents : int
        Number of inidividuals in data. The default is 100.
    n_questions : int
        number of questions each individual answered. The default is 10.
    scale_steps : int
        Scale, can be an Likert-scale. Always begins with 1 to 'scale_steps'.
        The default is 10.
    n_comp : int
        number of communities in data. The default is 2.
    mu_dist : float
        distance between the mu of the groups. Example: For n_comp=1,
        scale_steps=7 and mu_dist=1, mu (group 1)= 4.5 and mu (group 2) = 5.5.
        The default is 1.
    sd : float
        Standard Diviation for the Gauss-curve. The default is 0.5.

    Raises
    ------
    ValueError
        mu_distance must be not to high!

    Returns
    -------
    df : DataFrame
        Simulated Data with answers for all simulated items. also includes a party-affiliation variable
        and an ID

    var_names : list
        Names of Variables in the dataframe

    """
    
    # to recognize the error. if n_questions smaller, just random allocation of answers.
    if n_comp > n_questions:
        raise Exception("Number of questions must be higher than the number of components")

    if not len(split_up) == n_comp or not (sum(split_up) == 1):
        raise ValueError(f"length of split up must be ={n_comp} and have a sum of 1!")
    
    # column names
    column_names = ['person_id','party_aff']
    for i in range(n_questions):
        column_names.append('Q_' + str(i+1))   
    
    # mu to start is calculated
    mu_basis = mu_dist

    mu = (scale_steps+1)/2 - mu_dist * int(n_comp/2) + (n_comp%2 == 0)* mu_dist/2

    # answers for each group are computed; every individual from a group answers
    # a set of questions relatively equal (normaldistribution around a mu)
    options = {}
    for option in range(1, n_comp+1):
        if mu<1 or mu>scale_steps:
            raise ValueError(f'mu_dist={mu_dist} is badly chosen')
        # int cuts from 1.8 to 1; so 0.5 add to get round correct.
        temp = list(map(int, np.random.normal(mu,sd, 100000)+0.5))
        options[option] = [in_scale for in_scale in temp if in_scale>0 and in_scale<(scale_steps+1)]
        #print(options[option])
        mu += mu_basis

    data_dict = {}
    comp = 1
    change_questions = int(n_questions/n_comp)
    arrow = 2
    
    # setup for each individual:
    divide_individuals = split_up[comp-1] * n_agents
    for individual in range(1, n_agents+1):
        # all answers are unified distributed; individual answers all
        data_dict[individual] = [individual, comp] + [random.randint(1, scale_steps) for i in range(n_questions)]
        
        # range of numbers which are typical for the group; relatively same answers 
        items_to_change = list(range(arrow, arrow + change_questions))
        
        # all groups are typical in the same amount of numbers.
        # leftover questions stay uniformly distributed
        if len(items_to_change) < change_questions:
            pass
        else:
            # change answers to group typical!
            for item in items_to_change:
                data_dict[individual][item] = random.choice(options[comp])
        
        #print(items_to_change, data_dict[individual])
        
        # division of the groups, increases component

        if int(divide_individuals) == individual:
            comp +=1
            if comp-1 < len(split_up):
                divide_individuals += split_up[comp - 1] * n_agents
            arrow += change_questions
            
    # makes a DataFrame out of the dictionary and ads ID
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=column_names)
    df.index.name = 'person_id'    

    return df


def create_simulated_dataset(n_agents=200, n_questions=20, scale_steps=7, n_comp=4, sd=0.5):
    
    column_names = ['party_aff']
    for i in range(n_questions):
        column_names.append('Q_' + str(i+1))
    print(column_names)
    
    # possible values for 2 groups
    options = {}
    mu_basis = 1/(n_comp+1)
    mu = mu_basis

    for option in range(1, n_comp+1):

        temp = list(map(int, (np.random.normal(mu*scale_steps,sd, 10000))))

        options[option] = [in_scale for in_scale in temp if in_scale>0 and in_scale<(scale_steps+1)] 
        #print(options[option])
        mu += mu_basis
    
    data_dict = {}
    comp = 1
    change_questions = int(n_questions/n_comp)
    arrow = 1
    for individual in range(1, n_agents+1):
        data_dict[individual] = [comp] + [random.randint(1, scale_steps) for i in range(n_questions)]

        items_to_change = list(range(arrow, arrow + change_questions))
        
        if len(items_to_change) < change_questions:
            pass
        else:
            for item in items_to_change:
                data_dict[individual][item] = random.choice(options[comp]) *10
        
        if int((n_agents/n_comp) * comp) == individual:
            print(individual)
            comp +=1
            print(comp)
            arrow += change_questions

    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=column_names)
    df.index.name = 'ID'    

    return df
    

def get_wss(reduce_data_hc, d_matrix, gn=False, reduce_data=True):
    """
    Uses KMeans instead of HC
    :param reduce_data_hc:
    :param d_matrix:
    :param gn:
    :param reduce_data:
    :return:
    """
    # calculation off wss from d_matrix
    cluster_count = [2]  # only for 2 clusters
    clusters_hc_dict = dict()  # to save results of node in clusters
    wss = []
    # if d_matrix is dictionary --> calculating wss for GN algorithm
    if gn:
        label = 'GN'
        # Girman-Newman-Algorithm, d_matrix has sub matrices
        # for number of cluster in graph calculate wss
        for size in cluster_count:
            wss_gn_sum = 0
            kmeans = KMeans(n_clusters=1)
            for n in range(size):
                kmeans.fit_predict(d_matrix[size][n])
                wss_gn_sum += kmeans.inertia_
            nodes_component_list = []
            wss.append(wss_gn_sum)
            if size > 1 and G_dict:
                for node, component in G_dict[size].nodes(data='component'):
                    nodes_component_list.append(component)

                # calculate the silhouette brauche eine große dist_matrix
                if size == 2:
                    pos_del_elements = list(range(len(hc_dist_matrix)))
                    while reduce_data_hc:
                        pos_del_elements.remove(reduce_data_hc.pop())
                    # delete non-used elements
                    a = np.delete(hc_dist_matrix, pos_del_elements, axis=0)
                    hc_dist_matrix = np.delete(a, pos_del_elements, axis=1)
                # silhouette_score needs more than one cluster! otherwise ValueError
                #           silhouette_gn_avg = silhouette_score(hc_dist_matrix, nodes_component_list)
                #           print("For n_clusters =", size,
                #                 "The average silhouette_score is [GN]:", silhouette_gn_avg)
    else:
        label = 'HC'
        # Hierarchical clustering

        if reduce_data:
            pos_del_elements = list(range(len(d_matrix)))
            while reduce_data_hc:
                pos_del_elements.remove(reduce_data_hc.pop())
            # delete non-used elements
            a = np.delete(d_matrix, pos_del_elements, axis=0)
            d_matrix = np.delete(a, pos_del_elements, axis=1)

        for size in cluster_count:
            kmeans = KMeans(n_clusters=size)
            clusters_hc_dict[size] = kmeans.fit_predict(d_matrix)
            # silhouette_score needs more than one cluster! otherwise ValueError
            #if size > 1:
            #    silhouette_hc_avg = silhouette_score(d_matrix, clusters_hc_dict[size])
            #    print("For n_clusters =", size,
            #          "The average silhouette_score is [HC]:", silhouette_hc_avg)
            wss.append(kmeans.inertia_)
    return wss[0]


def random_one_var(wss, data_rescaled_matrix, rescaled_data_list, attitude_names, q):
    """
    Shuffles the one variable of the data and calculates WSS.

    :param wss: float; WSS without shuffling
    :param data_rescaled_matrix: np.stack; data matrix
    :param rescaled_data_list: list of data
    :param attitude_names: list of names; names of variables to plot
    :param q: Queue; Calculated WSS
    :return: None
    """
    row = []
    # select number and save in list
    select_var_pos = random.randint(0, len(rescaled_data_list[0])-1)

    row.append(attitude_names[select_var_pos])
    # get a shuffled data matrix for one variable
    shuffled_data_matrix = deepcopy(data_rescaled_matrix)

    random.shuffle(shuffled_data_matrix[:, select_var_pos])

    # calculate the new dist matrix
    shuffled_dist_matrix = d_matrix(shuffled_data_matrix)

    # get new WSS
    wss_shuffled, *unused = get_wss_HC(shuffled_data_matrix, shuffled_dist_matrix, n_clusters=2)
    row.append(wss_shuffled)

    # calculate distance between new and old wss
    wss_difference = (wss_shuffled - wss)/ wss
    row.append(wss_difference)

    # save it
    q.put(row)


def random_one_var_fixed_clusters(wss, data_rescaled_matrix, rescaled_data_list, attitude_names, component_list, q):
    """
    Shuffles the one variable of the data and calculates WSS.

    :param wss: list of float; WSS without shuffling [HC or GN or both]
    :param data_rescaled_matrix: np.stack; data matrix
    :param rescaled_data_list: list of data
    :param attitude_names: list of names; names of variables to plot
    :param q: Queue; Calculated WSS
    :return: None
    """
    row = []
    # select number and save in list
    select_var_pos = random.randint(0, len(rescaled_data_list[0])-1)

    # Add changed var name
    row.append(attitude_names[select_var_pos])
    # get a shuffled data matrix for one variable
    shuffled_data_matrix = deepcopy(data_rescaled_matrix)

    np.random.shuffle(shuffled_data_matrix[:, select_var_pos])

    # calculate the new dist matrix
    shuffled_dist_matrix = d_matrix(shuffled_data_matrix)

    # get new WSS, but with the same component list
    wss_shuffled, *unused = get_wss_HC(shuffled_data_matrix, shuffled_dist_matrix,
                                       n_clusters=2, component_list=component_list)
    row.append(wss_shuffled)

    # calculate distance between new and old wss
    wss_difference = (wss_shuffled - wss)/ wss
    row.append(wss_difference)

    # save it
    q.put(row)


def shuffled_both(wss, data_rescaled_matrix, rescaled_data_list, attitude_names, component_list, q):
    """
    Shuffles the one variable of the data and calculates WSS.

    :param wss: list of float; WSS without shuffling [HC or GN or both]
    :param data_rescaled_matrix: np.stack; data matrix
    :param rescaled_data_list: list of data
    :param attitude_names: list of names; names of variables to plot
    :param q: Queue; Calculated WSS
    :return: None
    """
    row = []
    # select number and save in list
    select_var_pos = random.randint(0, len(rescaled_data_list[0])-1)

    # Add changed var name
    row.append(attitude_names[select_var_pos])
    # get a shuffled data matrix for one variable
    shuffled_data_matrix = deepcopy(data_rescaled_matrix)
    
    random.shuffle(shuffled_data_matrix[:, select_var_pos])


    # calculate the new dist matrix
    shuffled_dist_matrix = d_matrix(shuffled_data_matrix)

    # calculate new wss for both
    for i in range(len(wss)):
        wss_shuffled, *unused = get_wss_HC(shuffled_data_matrix, shuffled_dist_matrix,
                                           n_clusters=2, component_list=component_list[i])
                                           

        row.append(wss_shuffled)
        # calculate distance between new and old wss
        wss_difference = (wss_shuffled - wss[i])/ wss[i]
        row.append(wss_difference)

    # save it
    q.put(row)


def Anont3_questions(data, rescaled_data_list, var_list, cluster_assigments, n_clusters=2, G_dict=None):
    """
    Does not change the cluster assignments!
    Get WSS. Shuffle the answer of one column. Repeat it 1000 times. Calculate the differences. Make a density plot for
    every variable in the graph.
    """
    print('relevant questions method', data, rescaled_data_list, var_list, cluster_assigments)

    attitude_names = var_list[2:]  # [2:] if party and ID are included
    names_pos_dict = dict()
    for i, name in enumerate(attitude_names):
        names_pos_dict[name] = i

    # calculate distance matrix for the selected data matrix
    data_rescaled_matrix = np.vstack(rescaled_data_list)

    dist_matrix = d_matrix(data_rescaled_matrix)

    reduce_data_hc = []
    if G_dict:
        wss, component_list = get_wss_HC(rescaled_data_list, dist_matrix, n_clusters=n_clusters, G=G_dict[n_clusters])
    else:
        wss, component_list = get_wss_HC(rescaled_data_list, dist_matrix, n_clusters=n_clusters)

    ##### just run it on the real components
    component_list = cluster_assigments

    results_shuffle_random_vars = []
    processes = []

    double = True
    if double and G_dict:
        wss_list = []
        component_list_list = []
        wss_hc, component_list_hc = get_wss_HC(rescaled_data_list, dist_matrix, n_clusters=n_clusters)
        wss_list.append(wss_hc)
        wss_list.append(wss)
        component_list_list.append(component_list_hc)
        component_list_list.append(component_list)
        count = 0
    for i in range(1):
        x = Queue()
        for _ in range(1):  # not more than 5; Queue receives to much info

            count += 1
            if double:
                p = Process(target=shuffled_both,
                            args=(wss_list, data_rescaled_matrix, rescaled_data_list, attitude_names, component_list_list, x))
            else:
                p = Process(target=random_one_var_fixed_clusters,
                            args=(wss, data_rescaled_matrix, rescaled_data_list, attitude_names, component_list, x))

            p.start()
            processes.append(p)
            # new_l.append(x.get()) # not working!
        for p in processes:
            p.join()
        while not x.empty():
            results_shuffle_random_vars.append(x.get())

    # var wss-shuffled-hc wss-difference-hc wss-shuffled-gn wss-shuffled
    # to integrate both methods on the same plot!
    results_split = []
    #print(results_shuffle_random_vars)
    for row in results_shuffle_random_vars:
        print(row)
        temp = [row[0], row[1], row[2], 'HC']
        results_split.append(temp)
        temp = [row[0], row[3], row[4], 'GN']
        #results_split.append(temp)

    df = pd.DataFrame.from_records(results_split, columns=['Variable', 'WSS', 'WSS-Distance', 'method'])

    print('do I get until here?', df)
    # print(df)
    #df['WSS-Distance'] = df['WSS-Distance']/df['WSS']
    # print(df)
    # prepare dataframe for return: mean() groupby()
    df = df.groupby(['Variable', 'method'])[['WSS-Distance']].mean()
    df = df.reset_index()

    # add random forest method and Boruta

    ###initialize Boruta
    forest = RandomForestRegressor(
        n_jobs=-1,  # using all processors
    )

    # HC
    boruta_hc = BorutaPy(
        estimator=forest,
        n_estimators='auto',
        max_iter=100  # number of trials to perform
    )
    ### fit Boruta (it accepts np.array, not pd.DataFrame)

        #short_data_list.append(rescaled_data_list[node-1])


    boruta_hc.fit(rescaled_data_list, component_list_hc)

    green_area = [attitude_names[i] for i, check in enumerate(boruta_hc.support_) if check]
    blue_area = [attitude_names[i] for i, check in enumerate(boruta_hc.support_weak_) if check]
    print('features in the green area:', green_area)
    print('features in the blue area:', blue_area)

    forest_hc = RandomForestClassifier()
    forest_hc.fit(rescaled_data_list, component_list_hc)
    rf_value_hc = forest_hc.feature_importances_

    boruta_hc_raking = boruta_hc.ranking_

    temp = []
    temp_boruta = []
    for Var, method in zip(df['Variable'], df['method']):
        if method == 'HC':
            temp.append(rf_value_hc[names_pos_dict[Var]])
            temp_boruta.append(boruta_hc_raking[names_pos_dict[Var]])
        else:  # GN
            print('not included in this type: is just method for the data set ANON T3')

    df["random_forest_value"] = temp
    df["boruta_ranking"] = temp_boruta

    plt.style.use('default')
    fig = plt.figure(figsize=(10, 7))
    fig.tight_layout()
    plt.grid(True)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.3)
    sns.violinplot(x="WSS-Distance", y="Variable", data=df, scale="width", palette="muted")
    #plt.savefig("violin_plot", pad_inches=2)
    plt.savefig('ANON_t3_violinplot.png')
    plt.close()
    #

    return df
    # compute pandas dataframe
    # plot and save density plot for variables


def relevant_questions_fixed_clusters(data, rescaled_data_list, var_list, n_clusters=2, G_dict=None):
    """
    Does not change the cluster assignments!
    Get WSS. Shuffle the answer of one column. Repeat it 1000 times. Calculate the differences. Make a density plot for
    every variable in the graph.
    """
    print('relevant questions method')

    attitude_names = var_list[2:]  # [2:] if party and ID are included
    names_pos_dict = dict()
    for i, name in enumerate(attitude_names):
        names_pos_dict[name] = i

    # calculate distance matrix for the selected data matrix
    data_rescaled_matrix = np.vstack(rescaled_data_list)

    dist_matrix = d_matrix(data_rescaled_matrix)

    reduce_data_hc = []
    if G_dict:
        wss, component_list = get_wss_HC(rescaled_data_list, dist_matrix, n_clusters=n_clusters, G=G_dict[n_clusters])
    else:
        wss, component_list = get_wss_HC(rescaled_data_list, dist_matrix, n_clusters=n_clusters)

    results_shuffle_random_vars = []
    processes = []

    double = True
    if double and G_dict:
        wss_list = []
        component_list_list = []
        wss_hc, component_list_hc = get_wss_HC(rescaled_data_list, dist_matrix, n_clusters=n_clusters)
        wss_list.append(wss_hc)
        wss_list.append(wss)
        component_list_list.append(component_list_hc)
        component_list_list.append(component_list)
    else:
      component_list_hc = component_list
    for i in range(5):
        x = Queue()
        for _ in range(3):  # not more than 5; Queue receives to much info
            if double and G_dict:
                p = Process(target=shuffled_both,
                            args=(wss_list, data_rescaled_matrix, rescaled_data_list, attitude_names, component_list_list, x))
            else:
                p = Process(target=random_one_var_fixed_clusters,
                            args=(wss, data_rescaled_matrix, rescaled_data_list, attitude_names, component_list, x))

            p.start()
            processes.append(p)
            # new_l.append(x.get()) # not working!
        for p in processes:
            p.join()
        while not x.empty():
            results_shuffle_random_vars.append(x.get())

    # var wss-shuffled-hc wss-difference-hc wss-shuffled-gn wss-shuffled
    # to integrate both methods on the same plot!
    results_split = []
    #print(results_shuffle_random_vars)
    for row in results_shuffle_random_vars:
        temp = [row[0], row[1], row[2], 'HC']
        results_split.append(temp)
        if G_dict:
            temp = [row[0], row[3], row[4], 'GN']
            results_split.append(temp)

    df = pd.DataFrame.from_records(results_split, columns=['Variable', 'WSS', 'WSS-Distance', 'method'])


    save_violinplot = True
    if save_violinplot:
        violin_df = df.copy(deep=True)

        #violin_df['Variable'] = violin_df['Variable'].astype(str) # has to be a string,  not working with int or float
        
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)
        fig = plt.figure(figsize=(10, 7))
        fig.tight_layout()
        plt.grid(True)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.3)
        violin_df.to_csv('violinplot_results.csv')
        #sns.violinplot(y="Variable", x="WSS-Distance", hue='method', data=violin_df, scale="width", palette="muted", split=True)
            #only hc
        #plt.savefig("violin_plotdouble", pad_inches=2)
        violin_df = violin_df.loc[violin_df['method']=='HC']
        #violin_df['Variable'] = violin_df['Variable'].astype('string') # has to be a string,  not working with int or float
        
        print(violin_df.info())
        plt.close()
        sns.violinplot(y="Variable", x="WSS-Distance", data=violin_df, scale="width", palette="Set3")
        plt.savefig("violin_plot")
        plt.close()
        sns.violinplot(y="WSS-Distance", x="Variable", data=violin_df, scale="width", palette="Set3")
        plt.savefig("violin_plot2", pad_inches=2)
        plt.close()
        sns.violinplot(x="WSS-Distance", y="Variable", data=violin_df, scale="width")
        plt.savefig("violin_plot3", pad_inches=2)
        plt.close()
        sns.violinplot(x="WSS-Distance", y="Variable", data=violin_df, palette="Set3")
        plt.savefig("violin_plot4", pad_inches=2)
        plt.close()
        planets = sns.load_dataset("planets")
        print(planets.info())
        ax = sns.violinplot(x="orbital_period", y="method",
                    data=planets[planets.orbital_period < 1000],
                    scale="width", palette="Set3")
        plt.savefig('violintest')
        #plt.show()
        #plt.close()
        # print(df)
        # prepare dataframe for return: mean() groupby()

    df = df.groupby(['Variable', 'method'])[['WSS-Distance']].mean()
    df = df.reset_index()

    # add random forest method and Boruta

    ###initialize Boruta
    forest = RandomForestRegressor(
        n_jobs=-1,  # using all processors
    )

    # HC
    boruta_hc = BorutaPy(
        estimator=forest,
        n_estimators='auto',
        max_iter=100  # number of trials to perform
    )
    ### fit Boruta (it accepts np.array, not pd.DataFrame)

    # in case we do not use all the information
    short_data_list = []

    # bring together
    new_data_dict = {}
    for row in data:
        new_data_dict[row[0]] = row[1:]


        #short_data_list.append(rescaled_data_list[node-1])

    boruta_hc.fit(rescaled_data_list, component_list_hc)

    forest_hc = RandomForestClassifier()
    forest_hc.fit(rescaled_data_list, component_list_hc)
    rf_value_hc = forest_hc.feature_importances_

    # GN
    if G_dict:
        # activate for reduced data size
        for i, node in enumerate(G_dict[2].nodes):
            short_data_list.append(new_data_dict[node])
        boruta_gn = BorutaPy(
            estimator=forest,
            n_estimators='auto',
            max_iter=100  # number of trials to perform
        )
        ### fit Boruta (it accepts np.array, not pd.DataFrame)
        boruta_gn.fit(np.array(short_data_list), component_list)

        forest_gn = RandomForestClassifier()
        forest_gn.fit(np.array(short_data_list), component_list)
        rf_value_gn = forest_gn.feature_importances_

    # classification from data
    boruta_data = BorutaPy(
        estimator=forest,
        n_estimators='auto',
        max_iter=100  # number of trials to perform
    )
    """
    ### fit Boruta (it accepts np.array, not pd.DataFrame)
    boruta_data.fit(rescaled_data_list, data['party_aff'].tolist())

    forest_real_class = RandomForestClassifier()
    print(data['party_aff'].tolist())
    forest_real_class.fit(rescaled_data_list, data['party_aff'].tolist())
    rf_value_realdata = forest_real_class.feature_importances_

    print(rf_value_realdata)
    print(df)
    df['rf_value_true_comp'] = [rf_value_realdata[names_pos_dict[i]] for i in df['Variable']]


    boruta_data_raking = boruta_data.ranking_
    df['boruta_value_true_comp'] = [boruta_data_raking[names_pos_dict[i]] for i in df['Variable']]
"""

    boruta_hc_raking = boruta_hc.ranking_
    if G_dict:
        boruta_gn_raking = boruta_gn.ranking_

    temp = []
    temp_boruta = []
    for Var, method in zip(df['Variable'], df['method']):
        if method == 'HC':
            temp.append(rf_value_hc[names_pos_dict[Var]])
            temp_boruta.append(boruta_hc_raking[names_pos_dict[Var]])
        else:  # GN
            temp.append(rf_value_gn[names_pos_dict[Var]])
            temp_boruta.append(boruta_gn_raking[names_pos_dict[Var]])

    df["random_forest_value"] = temp
    df["boruta_ranking"] = temp_boruta

    return df
    # compute pandas dataframe
    # plot and save density plot for variables




def relevant_questions(data_list, rescaled_data_list, var_list):
    """
    for 2
    Get WSS. Shuffle the answer of one column. Repeat it 1000 times. Calculate the differences. Make a density plot for
    every variable in the graph.
    """
    print('relevant questions method')

    attitude_names = var_list[2:]

    # calculate distance matrix for the selected data matrix
    data_rescaled_matrix = np.vstack(rescaled_data_list)

    dist_matrix = d_matrix(data_rescaled_matrix)
    # generate graph
    #G_dict, rescaled_features = get_graph(data_list, rescaled_data_list, max_number_cluster=2)
    # generate graphs distance matrix
    #sub_dist_matrix, reduce_data_hc, gn_graphs = run_gn_and_compute_dist_matrix(G_dict, data_list, dist_matrix)

    reduce_data_hc = []
    wss, *unused = get_wss_HC(rescaled_data_list, dist_matrix, n_clusters=2)

    results_shuffle_random_vars = []
    processes = []
    for i in range(200):
        x = Queue()
        for _ in range(5):  # not more than 5; Queue receives to much info
            p = Process(target=random_one_var, args=(wss, data_rescaled_matrix, rescaled_data_list, attitude_names, x))
            p.start()
            processes.append(p)
            # new_l.append(x.get()) # not working!
        for p in processes:
            p.join()
        while not x.empty():
            results_shuffle_random_vars.append(x.get())

    # to integrate both methods on the same plot!

    df = pd.DataFrame.from_records(results_shuffle_random_vars, columns=['Variable', 'WSS', 'WSS-Distance'])
    print(df)
    fig = plt.figure(figsize=(10, 7))
    fig.tight_layout()
    plt.grid(True)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.3)
    sns.violinplot(x="WSS-Distance", y="Variable", data=df, scale="width", palette="Set3")
    plt.savefig("violin_plot", pad_inches=2)

    # compute pandas dataframe
    # plot and save density plot for variables


def d_matrix(distance_vector, dist_fct='euclidean'):
    """
    Calculate the distance between every person A to person B and store data in a
    matrix = #people x #people
    --> calculate the Manhattan block distance
    :param distance_vector: np.vstack; distance matrix,
    :param dist_fct: string; default='euclidean'; which funktion to use to calculate the distance matrix
    :return: np.vstack; distance matrix
    """

    # Calculate distance matrix as a list
    dist_matrix = []
    for i in distance_vector:
        arr = []
        for j in distance_vector:
            if dist_fct == 'euclidean':
                dist_vec = ((i-j)**2)
            else:  # dist_fct == 'manhattan'
                dist_vec = abs(i-j)

            arr.append(dist_vec.sum())  # just sum up the abs(distance)
        dist_matrix.append(np.array(arr))

    # return  a distance matrix as a an array
    return np.vstack(dist_matrix)


def creating_dendogram_from_dmatrix(data_vstack, num_clusters=2):
    """
    computes and saves dendogram of distance matrix [HC]
    :param data_vstack: distance matrix
    :param num_clusters: default = 2;
    :return: None
    """
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram")
    dend = shc.dendrogram(shc.linkage(data_vstack, method='ward'))
    # crossline into dendogram: plt.axhline(y=6, color='r', linestyle='--')
    plt.savefig("test.png")


def get_wss_HC(rescaled_data_list, dist_matrix, n_clusters, G=None, component_list=None):
    """
    calculated SumSumSum (x - x_mean)**2
    :param rescaled_data_list: list of lists with rescaled items for each participant
    :param dist_matrix: calculate distance_matrix
    :param n_clusters: int; how many clusters in data
    :param G: nx.graph
    :param component_list: division of components, precalculated for comparing questions
    :return: wss for all components
    """
    wss_sum = 0

    # distance matrix has a size of 1, single cluster
    if len(dist_matrix) <= 1:
        return wss_sum, []

    if G:
        # mostly used for GN
        # get positions of participant for cluster X
        comp_dict = {}
        component_list = []

        # See which node is in which component
        for index, node_comp in enumerate(G.nodes(data='component')):
            comp = node_comp[1]  # node's component
            component_list.append(node_comp[1])
            if not comp_dict.get(comp):
                comp_dict[comp] = []
            comp_dict[comp].append(list(rescaled_data_list[index]))

        wss_comp = 0

        for matrix in comp_dict.values():
            # calculate mean
            sub_matrix = np.vstack(matrix)
            mean = []
            count = len(sub_matrix[:])

            for column in range(len(sub_matrix[0])):
                mean.append(sum(sub_matrix[:, column]) / count)
                              # calculate distance to mean for every participant and every question
            for row in sub_matrix:
                for i, item in enumerate(row):
                    wss_comp += (item - mean[i]) ** 2  # Calculate euklidean distance
        wss_sum += wss_comp
    else:
        #print('HC')
        # used for HC
        # NO COMPONENT_LIST? Do I have a component list from another WSS? Relevant for important questions  method
        if component_list is None:
            model = AgglomerativeClustering(n_clusters=n_clusters,
                                            affinity='euclidean',
                                            linkage='ward')
            # get positions of participant for cluster X
            component_list = model.fit_predict(dist_matrix)

        comp_dict = {}

        for index, comp in enumerate(component_list):
            if not comp_dict.get(comp):
                comp_dict[comp] = []
            comp_dict[comp].append(list(rescaled_data_list[index]))

        # calculate wss for each component
        wss_comp_dict = dict()  # not really used, saves the wss for each cluster
        #print(component_list, comp_dict)
        for key, comp in comp_dict.items():
            wss_comp = 0
            if len(comp) > 1:
                # calculate mean
                sub_matrix = np.vstack(comp)
                mean = []
                count = len(sub_matrix[:])
                for column in range(len(sub_matrix[0])):
                    mean.append(sum(sub_matrix[:, column]) / count)
                # calculate distance to mean for every participant and every question
                for row in sub_matrix:
                    for i, item in enumerate(row):
                        wss_comp += (item - mean[i]) ** 2  # Calculate euklidean distance
            wss_comp_dict[key] = wss_comp
            wss_sum += wss_comp

    return wss_sum, component_list


def elbow_wss(d_matrix, reduce_data_hc, max_amount_clusters, rescaled_data_list, mu_dist=-1, G_dict=None, hc_dist_matrix=None,
              plot_name=""):
    """
    Calculate the within square of sum for 1 to max amount of clusters. Generating elbow plot.
    For Hierarchical clustering OR Girvan-Newman method. Without KMeans, using for-loops to calculate
    method.
    Saves the plot.
    :param d_matrix: distance matrix
    :param reduce_data_hc: list;  names of nodes which are in the giant component
    :param max_amount_clusters: max number of clusters
    :return: dictionary of with clusters
    """
    # inner method became more import --> now get_wss_HC

    cluster_count = range(1, max_amount_clusters + 1)
    clusters_hc_dict = dict()  # to save results of node in clusters
    wss = []
    # if d_matrix is dictionary --> calculating wss for GN algorithm
    if isinstance(d_matrix, dict):
        label = 'GN' + str(round(mu_dist, 1))
        # Girman-Newman-Algorithm, d_matrix has sub matrices
        # for number of cluster in graph calculate wss
        for size in cluster_count:
            wss_gn_sum = 0

            wss_complete, unused = get_wss_HC(rescaled_data_list, d_matrix[size][0], 1, G=G_dict[size])
            wss_gn_sum += wss_complete
            nodes_component_list = []
            wss.append(wss_gn_sum)
            if size > 1 and G_dict:
                for node, component in G_dict[size].nodes(data='component'):
                    nodes_component_list.append(component)

                if size == 2:
                    pos_del_elements = list(range(len(hc_dist_matrix)))
                    while reduce_data_hc:
                        pos_del_elements.remove(reduce_data_hc.pop())
                    # delete non-used elements
                    a = np.delete(hc_dist_matrix, pos_del_elements, axis=0)
                    hc_dist_matrix = np.delete(a, pos_del_elements, axis=1)
                # silhouette_score needs more than one cluster! otherwise ValueError
                # silhouette_gn_avg = silhouette_score(hc_dist_matrix, nodes_component_list)
                # print("For n_clusters =", size,
                #      "The average silhouette_score is [GN]:", silhouette_gn_avg)
    else:
        label = 'HC'
        # Hierarchical clustering
        reduce_data = True

        if reduce_data:
            pos_del_elements = list(range(len(d_matrix)))
            while reduce_data_hc:
                pos_del_elements.remove(reduce_data_hc.pop())
            # delete non-used elements
            a = np.delete(d_matrix, pos_del_elements, axis=0)
            d_matrix = np.delete(a, pos_del_elements, axis=1)
        for size in cluster_count:
            wss_complete, clusters_hc_dict[size] = get_wss_HC(rescaled_data_list, d_matrix, size)
            # silhouette_score needs more than one cluster! otherwise ValueError
            # if size > 1:
            #    silhouette_hc_avg = silhouette_score(d_matrix, clusters_hc_dict[size])
            #    print("For n_clusters =", size,
            #          "The average silhouette_score is [HC]:", silhouette_hc_avg)
            wss.append(wss_complete)

    ### >>>>>>>plotting
    print(cluster_count, wss)
    elbow = KneeLocator(cluster_count, wss, curve="convex", direction="decreasing").knee
    if plot_name:
        plt.close()
        print(elbow)
        if elbow:
            plt.plot(cluster_count, wss, 'o', label=label, linestyle='--')
            plt.plot(cluster_count[elbow-1], wss[elbow-1], 'bx')
            plt.title('double elbow')
            plt.xlabel('Number of clusters')
            plt.ylabel('WSS')
            plt.legend()
            plt.savefig(f'WSS_test_{plot_name}')

    if G_dict:
        print('elbow', elbow)
        print('hehasdfsdf')
        print('hehasdfsdf')
        print('hehasdfsdf')
        print('hehasdfsdf')
        return int(elbow) if elbow else 2

    return clusters_hc_dict


def comparing_elbows(d_matrix, reduce_data_hc, max_amount_clusters, G_dict=None, hc_dist_matrix=None):
    """
    Calculate the within square of sum for 1 to max amount of clusters. Generating elbow plot.
    For Hierarchical clustering OR Girvan-Newman method.
    Saves the plot.
    :param d_matrix: distance matrix
    :param reduce_data_hc: list;  names of nodes which are in the giant component
    :param max_amount_clusters: max number of clusters
    :return: dictionary of with clusters
    """

    cluster_count = range(1, max_amount_clusters+1)
    clusters_hc_dict = dict()  # to save results of node in clusters
    wss = []
    # if d_matrix is dictionary --> calculating wss for GN algorithm
    if isinstance(d_matrix, dict):
        label = 'GN'
        # Girman-Newman-Algorithm, d_matrix has sub matrices
        # for number of cluster in graph calculate wss
        for size in cluster_count:
            wss_gn_sum = 0
            kmeans = KMeans(n_clusters=1)
            for n in range(size):
                kmeans.fit_predict(d_matrix[size][n])
                wss_gn_sum += kmeans.inertia_
            nodes_component_list = []
            wss.append(wss_gn_sum)
            if size > 1 and G_dict:
                for node, component in G_dict[size].nodes(data='component'):
                    nodes_component_list.append(component)

                # calculate the silhouette brauche eine große dist_matrix
                if size == 2:
                    pos_del_elements = list(range(len(hc_dist_matrix)))
                    while reduce_data_hc:
                        pos_del_elements.remove(reduce_data_hc.pop())
                    # delete non-used elements
                    a = np.delete(hc_dist_matrix, pos_del_elements, axis=0)
                    hc_dist_matrix = np.delete(a, pos_del_elements, axis=1)
                # silhouette_score needs more than one cluster! otherwise ValueError
                # silhouette_gn_avg = silhouette_score(hc_dist_matrix, nodes_component_list)
                # print("For n_clusters =", size,
                #      "The average silhouette_score is [GN]:", silhouette_gn_avg)
    else:
        label = 'HC'
        # Hierarchical clustering
        reduce_data = True

        if reduce_data:
            pos_del_elements = list(range(len(d_matrix)))
            while reduce_data_hc:
                pos_del_elements.remove(reduce_data_hc.pop())
            # delete non-used elements
            a = np.delete(d_matrix, pos_del_elements, axis=0)
            d_matrix = np.delete(a, pos_del_elements, axis=1)

        for size in cluster_count:
            kmeans = KMeans(n_clusters=size)
            clusters_hc_dict[size] = kmeans.fit_predict(d_matrix)
            # silhouette_score needs more than one cluster! otherwise ValueError
            #if size > 1:
            #    silhouette_hc_avg = silhouette_score(d_matrix, clusters_hc_dict[size])
            #    print("For n_clusters =", size,
            #          "The average silhouette_score is [HC]:", silhouette_hc_avg)
            wss.append(kmeans.inertia_)

    plt.plot(cluster_count, wss, label=label)
    plt.title('double elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WSS')
    plt.legend()
    plt.savefig('WSS_test')
    return clusters_hc_dict

# SEE The AgglomerativeClustering object performs a hierarchical clustering using a bottom up approach:
# each observation starts in its own cluster, and clusters are successively merged together.
# The linkage criteria determines the metric used for the merge strategy:


def run_gn_and_compute_dist_matrix(G_dict, data_list, dist_matrix):
    """ Run community detection method with Girman-Newman algorithm. Determine distance matrix for each
        community/component in graph
    :param G_dict: dictionary of nx.Graphs; generated graphs with different communities
    :param data_list: list; contains whole data set
    :param dist_matrix: np.vstack; distance matrix of the
    :param lines: int; first X lines of the Dataset
    :return: dict of np.vstack: distance matrices, one for each community,
             list: pos of nodes, which are not in biggest component [make it comparable to HC]
             nx.graph: Graph, which has only the biggest component as nodes
    """

    data_matrix = np.vstack(data_list)

    # save each sub_dist_matrix for each G
    sub_dist_matrix = dict()

    save_to_delete_pos_all = []
    for cluster_index, G in enumerate(G_dict.values(), start=1):
        # save names of nodes in dictionary for each cluster
        nodes = dict()
        for node, component in G.nodes(data='component'):
            if not nodes.get(component):
                nodes[component] = []
            nodes[component].append(node)

        # calculate the distance matrix for each component
        sub_dist_matrix[cluster_index] = dict()
        # list to save positions of IGNORED nodes (not part of biggest component)

        for key, node_names in nodes.items():
            save_pos = []
            index = 0
            # have to be sorted
            for node in sorted(node_names):
                while not (node == data_matrix[index][0]):
                    index += 1
                # only collecting from biggest component, first   --------------------now biggest cluster for cluster =2
                if cluster_index == 2:                      # for 1. biggest cluster "==1"
                    save_to_delete_pos_all.append(index)
                save_pos.append(index)
                index += 1

            # inverts the scale:
            del_list = list(range(len(data_matrix)))
            while save_pos:
                del_list.remove(save_pos.pop())

            # removing all rows and columns with nodes which are not part of biggest component
            sub_dist_matrix[cluster_index][key] = np.delete(dist_matrix, del_list, axis=0)
            sub_dist_matrix[cluster_index][key] = np.delete(sub_dist_matrix[cluster_index][key], del_list, axis=1)

    return sub_dist_matrix, save_to_delete_pos_all, G_dict


def hc_gn_confusion_matrix(G, predicted_values_hc, data_matrix):
    """
    Calculate normalized matrix for GN and HC to compare the two methods. Prints the confusion matrix of the two
    methods comparing the clustering results to the nodes party.

     Confusion Matrix
                                        event			   no-event
                        event       true positive       false positive
                     no-event       false negative      true negative

    :param G: nx.graph; with nodes, which have data='component' [0 or 1]
    :param predicted_values_hc: party of individuals; 1= democrat; 2= republican
    :param data_matrix: party and attitudes of every individual
    """

    # inner function for normalization of difusion matrix
    def normalize(con_mat):
        """
        Normalizes the confusion matrix
        :param con_mat: 2x2 confusion matrix
        :return None
        """
        # divides every values by their row-sum-value
        for row in con_mat:
            total = row.sum(axis=0)
            for i in range(len(row)):
                row[i] /= total

    # connect node name and party in a dictionary
    node_party = dict(zip(data_matrix[:, 0], data_matrix[:, 1]))
    expected_values = []
    predicted_values_gn = []

    # calculate expected and predicted party affiliation, only nodes which are Democrat OR Republican, else ignore
    for node, component in sorted(G.nodes('component')): # node rausschmeißen wenn node nicht zu einer party gehört
        if node_party[node] == 1 or node_party[node] == 2:
            dem_or_rep = 0 if node_party[node] == 1 else 1
            expected_values.append(dem_or_rep)
            predicted_values_gn.append(component)

    # Compute confusion matrix for Girman Newman cluster detection
    confusion_matrix_gn1 = np.array([[0., 0.], [0., 0.]])
    for expected, predicted in zip(expected_values, predicted_values_gn):
        confusion_matrix_gn1[expected][predicted] += 1

    # Numbering of clusters can change
    # Compute confusion matrix for Hierarchical clustering
    confusion_matrix_hc1 = np.array([[0., 0.], [0., 0.]])
    for expected, predicted in zip(expected_values, predicted_values_hc):
        confusion_matrix_hc1[expected][predicted] += 1

    confusion_matrix_hc_gn = np.array([[0., 0.], [0., 0.]])
    for expected, predicted in zip(predicted_values_hc, predicted_values_gn):
        confusion_matrix_hc_gn[expected][predicted] += 1



    #print("\nHC-confusion-matrix [HC vs. GN]")
    normalize(confusion_matrix_hc_gn)
    #print(confusion_matrix_hc_gn)

    #print("\nHC-confusion-matrix [HC vs. Party affiliation]")
    normalize(confusion_matrix_hc1)
    #print(confusion_matrix_hc1)

    #print('GN-confusion-matrix [GN vs. Party affiliation]')
    normalize(confusion_matrix_gn1)
    #print(confusion_matrix_gn1)

    # changing order of the expected_values to compare the two confusion matrices hc1 vs. hc2 ; gn1 vs. gn2;
    expected_values = []
    for node, component in sorted(G.nodes('component')):  # node rausschmeißen wenn node nicht zu einer party gehört
        if node_party[node] == 1 or node_party[node] == 2:
            dem_or_rep = 1 if node_party[node] == 1 else 0
            expected_values.append(dem_or_rep)

    # Compute confusion matrix for Girman Newman cluster detection
    confusion_matrix_gn2 = np.array([[0., 0.], [0., 0.]])
    for expected, predicted in zip(expected_values, predicted_values_gn):
        confusion_matrix_gn2[expected][predicted] += 1

    # Numbering of clusters can change
    # Compute confusion matrix for Hierarchical clustering
    confusion_matrix_hc2 = np.array([[0., 0.], [0., 0.]])
    for expected, predicted in zip(expected_values, predicted_values_hc):
        confusion_matrix_hc2[expected][predicted] += 1



    #print("\nHC-confusion-matrix [HC vs. Party affiliation]")
    normalize(confusion_matrix_hc2)
    #print(confusion_matrix_hc2)

    #print('GN-confusion-matrix [GN vs. Party affiliation]')
    normalize(confusion_matrix_gn2)
    #print(confusion_matrix_gn2)

    # compare the results of two matrices
    if confusion_matrix_gn1[0][0]+confusion_matrix_gn1[1][1] > confusion_matrix_gn2[0][0]+confusion_matrix_gn2[1][1]:
        confusion_matrix_gn = confusion_matrix_gn1
    else:
        confusion_matrix_gn = confusion_matrix_gn2

    # compare the results of two matrices
    if confusion_matrix_hc1[0][0]+confusion_matrix_hc1[1][1] > confusion_matrix_hc2[0][0]+confusion_matrix_hc2[1][1]:
        confusion_matrix_hc = confusion_matrix_hc1
    else:
        confusion_matrix_hc = confusion_matrix_hc2

    return confusion_matrix_gn, confusion_matrix_hc

def scikitlearn_confusion_matrix_compare_hc_ANES(gn_network, predicted_values, data_list,n_comp):
    """   ONLY ANES dataset2016
    confusion matrix is calculated with the scikitlearn-package
    :param gn_network: networkx-graph with data as nodes
    :param predicted_values: component assignment for each individual from HC (for test)
    :param data_list: has the data to the node to check the results against ground truth
    :param n_comp: number of components in the data
    """
    # connect node name and party in a dictionary
    node_party = predicted_values
    order = [i for i in range(n_comp)]

    #GN
    predicted_values_gn = []
    for node, component in sorted(gn_network.nodes('component')):
        predicted_values_gn.append(component)

    possible_permutations = list(list(itertools.permutations(order, n_comp)))

    result_confusion_matrix =np.zeros((n_comp, n_comp))
    test = 0
    for order in possible_permutations:
        new_node_party = []
        # address different labels for clusters to the nodes
        i = 0
        for node, component in sorted(gn_network.nodes('component')):
            new_node_party.append(order[node_party[i]])
            i += 1
        matrix = confusion_matrix(new_node_party, predicted_values_gn, labels=None) # normalise functioniert hier nciht
        cm = matrix / np.sum(matrix)
        
        # get biggest confusion matrix
        new_cm = 0
        result_communities = 'error'            # if it stays error, something went wrong
        for i in range(n_comp):
            new_cm += cm[i][i]
        if new_cm > test:
            result_confusion_matrix_hc_gn = cm
            test = new_cm
            # save result of components

    # save result of node party here, can be also the other way round, should just focus on number of components
    result_communities = new_node_party

    # integrate the ground thruth of the data and compare it to our results, which partymembers are in which component
    keys = [data_list[i][0] for i in range(len(data_list))]
    values = [data_list[i][1] for i in range(len(data_list))]
    party_affiliation_dict = dict(zip(keys, values))

    party_communities = {1: {'democrat': 0, 'republican': 0, 'unknown': 0},
                         0: {'democrat': 0, 'republican': 0, 'unknown': 0}
                         }

    for node, community in zip(gn_network.nodes(), result_communities):
        if party_affiliation_dict[node] == 1:
            party_communities[community]['democrat'] += 1
        elif party_affiliation_dict[node] == 2:
            party_communities[community]['republican'] += 1
        else:
            party_communities[community]['unknown'] += 1
    print(party_communities)
    return result_confusion_matrix_hc_gn, party_communities

def scikitlearn_confusion_matrix_compare_hc(gn_network, predicted_values_hc, n_comp):
    """
    confusion matrix is calculated with the scikitlearn-package
    :param gn_network: networkx-graph with data as nodes
    :param predicted_values_hc: component assignment for each individual from HC (for test)
    :param n_comp: number of components in the data
    """
    # connect node name and party in a dictionary
    node_party = predicted_values_hc
    order = [i for i in range(n_comp)]

    #GN
    predicted_values_gn = []
    for node, component in sorted(gn_network.nodes('component')):
        predicted_values_gn.append(component)

    possible_permutations = list(list(itertools.permutations(order, n_comp)))

    result_confusion_matrix = {'GN': np.zeros((n_comp, n_comp))}
    test = 0
    for order in possible_permutations:
        new_node_party = []
        # address different labels for clusters to the nodes
        i = 0
        for node, component in sorted(gn_network.nodes('component')):
            new_node_party.append(order[node_party[i]])
            i += 1

        matrix = confusion_matrix(new_node_party, predicted_values_gn, labels=None)  #, normalize='all')
        cm_hcgn = matrix / np.sum(matrix)

        # get biggest confusion matrix
        new_cm = 0
        for i in range(n_comp):
            new_cm += cm_hcgn[i][i]
        if new_cm > test:
            result_confusion_matrix_hc_gn = cm_hcgn
            test = new_cm
    print(result_confusion_matrix_hc_gn)
    return result_confusion_matrix_hc_gn


def scikitlearn_confusion_matrix_lead_eigen(gn_network, predicted_values_infomap, data_matrix, n_comp):
    """ Confusion matrix for infomap

    confusion matrix is calculated with the scikitlearn-package
    :param gn_network: networkx-graph with data as nodes
    :param predicted_values_hc: component assignment for each individual from HC
    :param data_matrix: component for each individual from data (test)
    :param n_comp: number of components in the data
    """
    # connect node name and party in a dictionary
    node_party = dict(zip(data_matrix[:, 0], data_matrix[:, 1]))

    order = [i for i in range(n_comp)]

    #GN
    predicted_values_gn = []
    for node, component in sorted(gn_network.nodes('component')):
        predicted_values_gn.append(component)

    possible_permutations = list(list(itertools.permutations(order, n_comp)))

    result_confusion_matrix = np.zeros((n_comp, n_comp))
    test = 0
    for order in possible_permutations:
        new_node_party = []

        # address different labels for clusters to the nodes
        for node, component in sorted(gn_network.nodes('component')):
            new_node_party.append(order[node_party[node]-1])

        cm = confusion_matrix(new_node_party, predicted_values_infomap, labels=None, normalize='all')

        # get biggest confusion matrix
        new_cm = 0
        for i in range(n_comp):
            new_cm += cm[i][i]
        if new_cm > test:
            result_confusion_matrix = cm
            test = new_cm

    return result_confusion_matrix

def scikitlearn_confusion_matrix_all_comdetec_ANES(gn_network, predicted_communities_alogs_dict, data_matrix, n_comp):
    """
    confusion matrix is calculated with the scikitlearn-package
    :param gn_network: networkx-graph with data as nodes
    :param predicted_communities_alogs_dict: component assignment for each individual from different community algos
    :param data_matrix: component for each individual from data (test)
    :param n_comp: number of components in the data
    """
    # Correct node allocation
    predicted_communities = deepcopy(predicted_communities_alogs_dict)

    node_party = dict(zip(data_matrix[:, 0], data_matrix[:, 1]))

    select_only_ground_truth = []
    temp_list = []
    for community in node_party:
        check = True if community==1 or community==2 else False
        select_only_ground_truth.append(check)
        if check:
            temp_list.append(community)
    node_party = temp_list # list shortend for ANES!!

    order = [i for i in range(n_comp)]
    # compute possible combinations from order!
    possible_permutations = list(list(itertools.permutations(order, n_comp)))
    if gn_network:
        # add GN to algorithms
        predicted_values_gn = []
        save_predicted_values_gn_for_dict = []
        for node, check in zip(sorted(gn_network.nodes('component')), select_only_ground_truth):
            save_predicted_values_gn_for_dict.append(node[1])  # node[1] = COMPONENT
            if check:
                predicted_values_gn.append(node[1])  #node[1] = COMPONENT
        predicted_communities_alogs_dict['GN'] = save_predicted_values_gn_for_dict
        predicted_communities['GN'] = predicted_values_gn

        # names of community detec algorithms
        cd_algorithms_names = [key for key in predicted_communities.keys()]

        for k,v in predicted_communities.items():
            temp_list = []
            for component, check in zip(v, select_only_ground_truth):
                if check:
                    temp_list.append(component)
            predicted_communities[k] = temp_list

        result_confusion_matrix = {}
        test = {}
        for name in cd_algorithms_names:
            result_confusion_matrix[name] = np.zeros((n_comp, n_comp))
            test[name] = 0

        for order in possible_permutations:
            new_node_party = []

            # address different labels for clusters to the nodes

            for i in range(len(predicted_values_gn)):
                print(node_party[i])
                new_node_party.append(order[node_party[i]-1])
            cm = {}
            for algo, prediction in predicted_communities.items():
                matrix = confusion_matrix(new_node_party, prediction, labels=None)  #, normalize='all')
                cm[algo] = matrix / np.sum(matrix)

            # get biggest confusion matrix
            for method in cd_algorithms_names:
                new_cm = 0
                for i in range(n_comp):
                    new_cm += cm[method][i][i]
                if new_cm > test[method]:
                    result_confusion_matrix[method] = cm[method]
                    test[method] = new_cm
    else:
        # names of community detec algorithms
        cd_algorithms_names = [key for key in predicted_communities.keys()]

        result_confusion_matrix = {}
        test = {}
        for name in cd_algorithms_names:
            result_confusion_matrix[name] = np.zeros((n_comp, n_comp))
            test[name] = 0

        for order in possible_permutations:
            new_node_party = []

            # address different labels for clusters to the nodes
            for node in node_party:
                new_node_party.append(order[node-1])
            cm = {}
            for algo, prediction in predicted_communities.items():
                matrix = confusion_matrix(new_node_party, prediction, labels=None)  #, normalize='all')
                cm[algo] = matrix / np.sum(matrix)

            # get biggest confusion matrix
            for method in cd_algorithms_names:
                new_cm = 0
                for i in range(n_comp):
                    new_cm += cm[method][i][i]
                if new_cm > test[method]:
                    result_confusion_matrix[method] = cm[method]
                    test[method] = new_cm
    print(result_confusion_matrix)
    return result_confusion_matrix

def scikitlearn_confusion_matrix_all_comdetec(gn_network, predicted_communities_alogs_dict, data_matrix, n_comp):
    """
    confusion matrix is calculated with the scikitlearn-package
    :param gn_network: networkx-graph with data as nodes
    :param predicted_communities_alogs_dict: component assignment for each individual from different community algos
    :param data_matrix: component for each individual from data (test)
    :param n_comp: number of components in the data
    """
    # Correct node allocation
    node_party = dict(zip(data_matrix[:, 0], data_matrix[:, 1]))
    order = [i for i in range(n_comp)]
    # compute possible combinations from order!
    possible_permutations = list(list(itertools.permutations(order, n_comp)))
    if gn_network:
        # add GN to algorithms
        predicted_values_gn = []
        for node, component in sorted(gn_network.nodes('component')):
            predicted_values_gn.append(component)
        predicted_communities_alogs_dict['GN'] = predicted_values_gn

        # names of community detec algorithms
        cd_algorithms_names = [key for key in predicted_communities_alogs_dict.keys()]

        result_confusion_matrix = {}
        test = {}
        for name in cd_algorithms_names:
            result_confusion_matrix[name] = np.zeros((n_comp, n_comp))
            test[name] = 0

        for order in possible_permutations:
            new_node_party = []

            # address different labels for clusters to the nodes

            for node, component in sorted(gn_network.nodes('component')):
                new_node_party.append(order[node_party[node]-1])
            cm = {}
            for algo, prediction in predicted_communities_alogs_dict.items():
                matrix = confusion_matrix(new_node_party, prediction, labels=None)  #, normalize='all')
                cm[algo] = matrix / np.sum(matrix)

            # get biggest confusion matrix
            for method in cd_algorithms_names:
                new_cm = 0
                for i in range(n_comp):
                    new_cm += cm[method][i][i]
                if new_cm > test[method]:
                    result_confusion_matrix[method] = cm[method]
                    test[method] = new_cm
    else:
        # names of community detec algorithms
        cd_algorithms_names = [key for key in predicted_communities_alogs_dict.keys()]

        result_confusion_matrix = {}
        test = {}
        for name in cd_algorithms_names:
            result_confusion_matrix[name] = np.zeros((n_comp, n_comp))
            test[name] = 0

        for order in possible_permutations:
            new_node_party = []

            # address different labels for clusters to the nodes
            for node in node_party.values():
                new_node_party.append(order[node-1])
            cm = {}
            for algo, prediction in predicted_communities_alogs_dict.items():
                matrix = confusion_matrix(new_node_party, prediction, labels=None)  #, normalize='all')
                cm[algo] = matrix / np.sum(matrix)

            # get biggest confusion matrix
            for method in cd_algorithms_names:
                new_cm = 0
                for i in range(n_comp):
                    new_cm += cm[method][i][i]
                if new_cm > test[method]:
                    result_confusion_matrix[method] = cm[method]
                    test[method] = new_cm
    print(result_confusion_matrix)
    return result_confusion_matrix



def scikitlearn_confusion_matrix(gn_network, predicted_values_hc, data_matrix, n_comp):
    """
    confusion matrix is calculated with the scikitlearn-package
    :param gn_network: networkx-graph with data as nodes
    :param predicted_values_hc: component assignment for each individual from HC
    :param data_matrix: component for each individual from data (test)
    :param n_comp: number of components in the data
    """
    # connect node name and party in a dictionary
    node_party = dict(zip(data_matrix[:, 0], data_matrix[:, 1]))

    order = [i for i in range(n_comp)]

    #GN
    predicted_values_gn = []
    for node, component in sorted(gn_network.nodes('component')):
        predicted_values_gn.append(component)

    possible_permutations = list(list(itertools.permutations(order, n_comp)))

    result_confusion_matrix = {'HC': np.zeros((n_comp, n_comp)), 'GN': np.zeros((n_comp, n_comp))}
    test = {'HC': 0, 'GN': 0}
    for order in possible_permutations:
        new_node_party = []

        # address different labels for clusters to the nodes
        for node, component in sorted(gn_network.nodes('component')):
            new_node_party.append(order[node_party[node]-1])

        print('########################################')
        print(new_node_party)
        print(predicted_values_gn)
        print('########################################')
        
        
        matrix_hc = confusion_matrix(new_node_party, predicted_values_hc, labels=None)
        matrix_gn = confusion_matrix(new_node_party, predicted_values_gn, labels=None)
        cm = {'HC': matrix_hc / np.sum(matrix_hc), #, normalize='all'),
              'GN': matrix_gn / np.sum(matrix_gn)} # , normalize='all')

        # get biggest confusion matrix
        for method in ['HC', 'GN']:
            new_cm = 0
            for i in range(n_comp):
                new_cm += cm[method][i][i]
            if new_cm > test[method]:
                result_confusion_matrix[method] = cm[method]
                test[method] = new_cm

    return result_confusion_matrix


def hc_gn_multi_confusion_matrix(G, predicted_values_hc, data_matrix, n_comp):
    """
    Calculate normalized matrix for GN and HC to compare the two methods. Prints the confusion matrix of the two
    methods comparing the clustering results to the nodes party.

     Confusion Matrix
                                        event			   no-event
                        event       true positive       false positive
                     no-event       false negative      true negative

    :param G: nx.graph; with nodes, which have data='component' [0 or 1]
    :param predicted_values_hc: party of individuals; 1= democrat; 2= republican
    :param data_matrix: party and attitudes of every individual
    """

    # inner function for normalization of difusion matrix
    def normalize(con_mat):
        """
        Normalizes the confusion matrix
        :param con_mat: 2x2 confusion matrix
        :return None
        """
        # divides every values by their row-sum-value
        total = len(predicted_values_hc)
        for row in con_mat:
            for i in range(len(row)):
                row[i] /= total

    # connect node name and party in a dictionary
    node_party = dict(zip(data_matrix[:, 0], data_matrix[:, 1]))
    predicted_values_gn = []

    order = [i for i in range(n_comp)]
    possible_permutations = list(list(itertools.permutations(order, n_comp)))

    for node, component in sorted(G.nodes('component')):
        predicted_values_gn.append(component)

    permutation_expected_values = {}
    permutation_confusion_matrix_gn = {}
    permutation_confusion_matrix_hc = {}
    for order in possible_permutations:
        print("in order", order)
        permutation_expected_values[order] = []

        for node, component in sorted(G.nodes('component')):
            permutation_expected_values[order].append(order[node_party[node]-1])

        permutation_confusion_matrix_gn[order] = np.zeros((n_comp, n_comp))
        permutation_confusion_matrix_hc[order] = np.zeros((n_comp, n_comp))

        for expected, predicted_gn, predicted_hc in zip(permutation_expected_values[order], predicted_values_gn, predicted_values_hc):
            permutation_confusion_matrix_gn[order][expected][predicted_gn] += 1
            permutation_confusion_matrix_hc[order][expected][predicted_hc] += 1
        print(permutation_confusion_matrix_gn[order])
        normalize(permutation_confusion_matrix_gn[order])
        normalize(permutation_confusion_matrix_hc[order])

    print(permutation_confusion_matrix_gn)
    # get best confusion matrix
    max_list_gn = []
    max_list_hc = []
    order_list = []
    for order, confusion_mx_gn, confusion_mx_hc in zip(permutation_confusion_matrix_gn.keys(), permutation_confusion_matrix_gn.values(), permutation_confusion_matrix_hc.values()):
        max_matrix_gn = 0
        max_matrix_hc = 0
        for i in range(n_comp):
            max_matrix_gn += confusion_mx_gn[i][i]
            max_matrix_hc += confusion_mx_hc[i][i]
        max_list_gn.append(max_matrix_gn)
        max_list_hc.append(max_matrix_hc)
        order_list.append(order)
    max_order_gn = order_list[max_list_gn.index(max(max_list_gn))]
    print(permutation_confusion_matrix_gn[max_order_gn])
    max_order_hc = order_list[max_list_hc.index(max(max_list_hc))]
    print(permutation_confusion_matrix_hc[max_order_hc])

    return permutation_confusion_matrix_gn[max_order_gn], permutation_confusion_matrix_hc[max_order_hc]


def generate_matrix_from_csv(csv_file, var_names, first_attitude_pos=0, remove_lines=[99, 'negative'],
                             read_until=None, anes2016=None, only_dem_rep_anes=None, shuffle_data=None):
    """ Filter data and generate a vstack from the Data, where the rows correspond to a number of people, who answered
    a number of questions, which are stored at as columns => Matrix: #people x #questions
    Select data from csv.-file and return a list
    :param csv_file: string
    :param first_attitude_pos; int; has to be adjusted for different data sets
    :param var_names: list of strings; used variables
    :param remove_lines: list; default=[99, 'negative']; values, which should be ignore/erased in data
    :param read_until: int; default=None -> read max data length
    :return: list
    """
    # ['V160001', 'V161155', 'V161232', 'V161189', 'V161192', 'V161209', 'V161231', 'V161201', 'V161187']

    # Automatically chekc the delimter of the csv-file
    check_csv = open(csv_file)
    dialect = csv.Sniffer().sniff(check_csv.readline(), delimiters='|;,')

    complete_data = pd.read_csv(csv_file, sep=dialect.delimiter, low_memory=False)
    
    # if activated, only integrate the democrats and the republicans
    if only_dem_rep_anes:
        complete_data = complete_data[(complete_data[only_dem_rep_anes] == 1) | (complete_data[only_dem_rep_anes] == 2)]
    
    selected_data = complete_data[var_names].copy()
    short_selected_data = selected_data.iloc[:read_until, :]  # without patriot flag
    data_list = short_selected_data.values.tolist()

    #print(selected_data)
    # remove negative values from data list:  after length = 4348 values
    lines_to_remove = []
    for line in data_list:
        line_attributes = line[first_attitude_pos:]
        check = False
        # special for ANES data, have to turned of for other data!
        if anes2016 and line_attributes[0] == 5:
            check = True
        for i in line_attributes:
            if i == " ":
                check = True
                break
            i = int(i)
            if i in remove_lines:
                check = True
                break
            if 'negative' in remove_lines and i < 0:
                check = True
                break
            if 'positive' in remove_lines and i > 0:
                check = True
                break
        if check:
            lines_to_remove.append(line)

    unvalid = len(lines_to_remove)

    # remove all collected negative lines from data
    while lines_to_remove:
        data_list.remove(lines_to_remove.pop())

    print(f'2. {unvalid} lines erased. Number of valid rows in data: {len(data_list)}')

    data = []
    if data_list:
        # ERROR correction: int has a limited size; like this only the answers of data are cast to int
        for values in data_list:
            data.append(values[:first_attitude_pos] + list(np.array(values[first_attitude_pos:]).astype(np.int)))

    return data


def frequency_table_relabel(G, predicted_values_hc):

    # calculate expected and predicted party affiliation, only nodes which are Democrat OR Republican, else ignore
    counter = dict()
    counter_pair = dict()
    component_values_gn = []
    for node_component, hc in zip(sorted(G.nodes('component')), predicted_values_hc):
        component_values_gn.append(node_component[1])
        if not counter.get((node_component[1], hc)):
            counter[(node_component[1], hc)] = 0
            counter_pair[(hc, node_component[1])] = 0
            counter_pair[(node_component[1], hc)] = 0
        counter[(node_component[1], hc)] += 1
        counter_pair[(hc, node_component[1])] += 1
        counter_pair[(node_component[1], hc)] += 1

    for key, value in sorted(counter.items(), key=lambda item: item[1], reverse=True):
        print(key, '--->', value)
    print('\n\nNEWWWW')

    for key, value in sorted(counter_pair.items(), key=lambda item: item[1], reverse=True):
        print(key, '--->', value)

    max_tuple = max(counter, key=counter.get)
    """
    for i, comp_name_gn in enumerate(component_values_gn):
        if comp_name_gn == max_tuple[0]:
            component_values_gn[i] = max_tuple[1]

    counter = dict()
    for comp_gn, hc in zip(component_values_gn, predicted_values_hc):
        if not counter.get((comp_gn, hc)):
            counter[(comp_gn, hc)] = 0
        counter[(comp_gn, hc)] += 1

    print('\n\nNEWWWW')
    for key, value in counter.items():
        print(key, '--->', value)
    """


def rescaling_data(data_to_rescale, scale_dict=None, scale_length_list=None):
    """
    Rescale data to fit in -1 and 1
    :param data_to_rescale: np.vstack;  distance vector
    :param scale_dict: predefined dictionary with scales
    :param scale_length_list: size of each scale
    :return: np.vstack; rescaled distance vector
    """
    # 19_08_

    if scale_dict:
        attribute_list = []
        for key in scale_dict.keys():
            attribute_list.append(key)

        # define POS for Variables
        var_start_pos = len(attribute_list)
    else:
        var_start_pos = len(scale_length_list)

    list_temp = []
    data_to_rescale = np.vstack(data_to_rescale)
    only_item_data = data_to_rescale[:, -var_start_pos:]  # last positions is reserved for variables
    print(only_item_data)
    for node, row in enumerate(only_item_data):
        sublist = []
        for column, value in enumerate(row):
            column = int(column)
            value = int(value)
            # if special rescaling is given
            if scale_dict:
                sublist.append(scale_dict[attribute_list[column]][value-1])
            # not, just recalculate
            else:
                midpoint = (scale_length_list[column] + 1) / 2
                sublist.append((value - midpoint) / (scale_length_list[column] - midpoint))
        list_temp.append(sublist)
    return np.array(list_temp)


def hc_confusion_matrix(predicted_values_hc, data_matrix, split_up, n_comp):
    """
    Calculate normalized matrix for ONLY HC to analyse goodness.

     Confusion Matrix
                                        event			   no-event
                        event       true positive       false positive
                     no-event       false negative      true negative

    :param predicted_values_hc: party of individuals; 1= democrat; 2= republican
    :param split_up:
    :param data_matrix: party and attitudes of every individual
    """

    # inner function for normalization of difusion matrix
    def normalize(con_mat):
        """
        Normalizes the confusion matrix
        :param con_mat: 2x2 confusion matrix
        :return None
        """
        # divides every values by their row-sum-value
        for row in con_mat:
            total = row.sum(axis=0)
            for i in range(len(row)):
                row[i] /= total

    # connect node name and party in a dictionary
    node_party = dict(zip(data_matrix[:, 0], data_matrix[:, 1]))

    order = [i for i in range(n_comp)]
    possible_permutations = list(list(itertools.permutations(order, n_comp)))

    permutation_expected_values = {}
    permutation_confusion_matrix_hc = {}

    for order in possible_permutations:
        # standard

        permutation_expected_values[order] = []

        for node in range(1, 101):
            permutation_expected_values[order].append(order[node_party[node]-1])

        permutation_confusion_matrix_hc[order] = np.zeros((n_comp, n_comp))

        for expected, predicted_hc in zip(permutation_expected_values[order], predicted_values_hc):
            permutation_confusion_matrix_hc[order][expected][predicted_hc] += 1
        normalize(permutation_confusion_matrix_hc[order])

    # get best confusion matrix
    max_list_hc = []
    order_list = []
    for order, confusion_mx_hc in zip(permutation_confusion_matrix_hc.keys(), permutation_confusion_matrix_hc.values()):
        max_matrix_hc = 0
        for i in range(n_comp):
            max_matrix_hc += confusion_mx_hc[i][i]
        max_list_hc.append(max_matrix_hc)
        order_list.append(order)
    max_order_hc = order_list[max_list_hc.index(max(max_list_hc))]
    # print(permutation_confusion_matrix_hc[max_order_hc])

    return permutation_confusion_matrix_hc[max_order_hc]