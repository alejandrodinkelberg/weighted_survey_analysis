import numpy as np
import random
import pandas as pd 
from scipy import stats
from sklearn.neighbors import KernelDensity
from ipfn import ipfn
from copy import deepcopy
from ANES_HierClus import rescaling_data
import ANES_com_detec_remodel_biggest_component

def run_data_to_graph_to_community(data_list=False, resplit=False, n_items=10):
    """[summary]

    Args:
        data_list (bool, optional): [description]. Defaults to False.
        resplit (bool, optional): [description]. Defaults to False.
        n_items (int, optional): [description]. Defaults to 10.

    Raises:
        ValueError: [description]
    """
    # Rescale the generated data from the synthetic data
    print(data_list)
    rescaled_data_list = rescaling_data(data_list, scale_length_list=['size']*n_items)
    
    
    gn_graph_dict, rescaled_features, threshold, edges_between_comp_renamed = ANES_com_detec_remodel_biggest_component.get_graph(
        data_list,
        rescaled_data_list,
        max_number_cluster=1,
        minimum_size_for_two_comp=resplit,
        # threshold=7.0
    )
    
    # define as dict from list to find it easier
    participants_dict = {line[0]:line[1:] for line in data_list}
    
    
    # Get the Republican and the democrats of the sample
    Rep_community = []
    Dem_community = []
    for i in zip(gn_graph_dict[1].nodes()):
        
        # Check whether it is Republican or Democrat
        
        # from tuple to  int
        node = i[0]
        
        if participants_dict.get(node)[0] == 1: # is defined in first argument of eb_list
            Dem_community.append(node)
        elif participants_dict.get(node)[0] == 2:
            Rep_community.append(node)
        else:
            raise ValueError("Something went wrong, neither a rep or dem?")
    
    # Calculate the Score of polarisation:
    results_dict = Rep_Dem_polarisation_score(gn_graph_dict[1], Rep_community, Dem_community)
    
    # Structure and save results
    df = pd.DataFrame(columns=[
        'seed',
        'threshold',
        'size',
        'size_rep_community',
        'size_dem_community',
        'cut_EB_length',
        'Rep_EB_list_length',
        'Dem_EB_list_length',
        'cut_EB_average',
        'Rep_EB_list_average',
        'Dem_EB_list_average',
        'cut_EB_variance',
        'Rep_EB_list_variance',
        'Dem_EB_list_variance',
        'mean-cut-divided-mean-rest',
        'entropy',
        'modularity score'
        ]
                 )
    
    results_dict['seed'] = seed
    results_dict['threshold'] = threshold
    results_dict['size'] = len(Rep_community) + len(Dem_community)
    results_dict['size_rep_community'] = len(Rep_community)
    results_dict['size_dem_community'] = len(Dem_community)
    results_dict['modularity score'] = nx_comm.modularity(gn_graph_dict[1], [Dem_community, Rep_community])

    df = df.append(results_dict, ignore_index=True)
    
    # save
    my_path = f'REP_DEM_anes{anes}_EB-measurements.csv'
    header = False if os.path.isfile(my_path) else True 
    df.to_csv(my_path, mode='a', header=header, index=False)
    
def Rep_Dem_polarisation_score(graph_nx, Rep_community, Dem_community):
        
    # dict_edgebetweenness: dictionary with all edge_betweenness calculation! ['node1,node2'] = EB
    dict_edgebetweenness = {}

    # set edge with edge bet
    edge_betweenness = nx.edge_betweenness_centrality(graph_nx, normalized=True)                    # normalised the edge betweenness score
    nx.set_edge_attributes(graph_nx, edge_betweenness, "betweenness")

    # get edge betweenness of links between Rep and Dems
    eb_list = []
    for node in Dem_community:
        for neighbor in Rep_community:
            if graph_nx.has_edge(node, neighbor):
                eb_list.append(graph_nx.edges[node, neighbor]["betweenness"])
    
    print('total', graph_nx.number_of_edges())
    print('eb_list', len(eb_list))
    print('Number of edges in Dem_community', graph_nx.subgraph(set(Dem_community)).size())# for block in ))
    print('Number of edges in Rep_community', graph_nx.subgraph(set(Rep_community)).size())
    
    # Democrats     Subgraph
    Dem_community_subgraph = graph_nx.subgraph(set(Dem_community))
    Dem_EB_list = []
    for edge in Dem_community_subgraph.edges(data=True):
        Dem_EB_list.append(edge[2]['betweenness'])
    
    # Republicans   Subgraph
    Rep_community_subgraph = graph_nx.subgraph(set(Rep_community))
    Rep_EB_list = []
    for edge in Rep_community_subgraph.edges(data=True):
        Rep_EB_list.append(edge[2]['betweenness'])

    print(np.mean(Rep_EB_list), np.mean(Dem_EB_list))
    print(np.mean(eb_list))

    # put into array and calculate mean and variance
    eb_array = np.asarray(eb_list)
    variance = np.var(eb_array)
    mean = np.mean(eb_array)

    # get edge betweenness from alll but not from cut
    eb_list_all = Rep_EB_list + Dem_EB_list

    print(len(eb_list), 'length of eblistall-->', len(eb_list_all), "\n")

    
    # not needed uses stats.entropy instead as it is normalized
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    
    # kernel density estimation --> probability distribution function --> KL-divergence    
    p = np.array(eb_list).reshape(-1, 1)
    q = np.array(eb_list_all).reshape(-1, 1)

    p_kde = KernelDensity(kernel='gaussian').fit(p)
    q_kde = KernelDensity(kernel='gaussian').fit(q)
    p = stats.norm.pdf(p_kde.sample(10000))
    q = stats.norm.pdf(q_kde.sample(10000))
    kl_div = kl_divergence(q, p)
    print('KL_divergence---->', kl_div)
    
    entropy_cut_vs_all = stats.entropy(p, q)[0]
    print(f'entropy----> {entropy_cut_vs_all}')

    results_dict = {
        'cut_EB_length': len(eb_list),
        'Rep_EB_list_length': len(Rep_EB_list),
        'Dem_EB_list_length': len(Dem_EB_list),
        'cut_EB_average': np.mean(eb_list),
        'Rep_EB_list_average': np.mean(Rep_EB_list),
        'Dem_EB_list_average': np.mean(Dem_EB_list),
        'cut_EB_variance': np.var(eb_list),
        'Rep_EB_list_variance': np.var(Rep_EB_list),
        'Dem_EB_list_variance': np.var(Dem_EB_list),
        'mean-cut-divided-mean-rest': np.mean(eb_list) / (np.mean(Rep_EB_list + Dem_EB_list)),
        'entropy': entropy_cut_vs_all,
    }
    
    return results_dict

def _create_sim_dataset_ranking(n_agents=100, n_questions=10, scale_steps=10, higher_sd_q=0, random_q=0, n_comp=2, mu_dist=1, sd=0.5, split_up=[0.5, 0.5]):
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


def generate_synth_attitude_surveys(n_agents=100, n_questions=10, scale_steps=10, mu_max=5, number_ranks=5, n_comp=2,
                                        sd=0.5, split_up=[0.5, 0.5], age=False):
    """ Generates different attitude surveys to play around with. The mu differences between groups is getting smaller for less important questions
        making it more dificult for to get the information about how to separate different groups
        

    Args:
        n_agents (int, optional): [description]. Defaults to 100.
        n_questions (int, optional): [description]. Defaults to 10.
        scale_steps (int, optional): [description]. Defaults to 10.
        mu_max (int, optional): [description]. Defaults to 5.
        number_ranks (int, optional): [description]. Defaults to 5.
        n_comp (int, optional): [description]. Defaults to 2.
        sd (float, optional): [description]. Defaults to 0.5.
        split_up (list, optional): [description]. Defaults to [0.5, 0.5].

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

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
        data_columns = _create_sim_dataset_ranking(n_agents, q_per_mu, scale_steps, mu_dist=mu_dist, n_comp=n_comp,
                                                  higher_sd_q=0, random_q=0, sd=sd, split_up=split_up)

        end = item_name+q_per_mu
        for name_id in range(item_name, end):
            name = "Q_" + str(name_id)
            names.append(name)
            data_set[name] = data_columns[list(data_columns.columns)[name_id-end]]

        item_name += q_per_mu

    random_questions = n_questions - sum(number_of_questions)

    random_data = _create_sim_dataset_ranking(n_agents, random_questions, scale_steps, mu_dist=mu_dist, n_comp=n_comp,
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
    
    if age:
        population = ['18-30', '31-50', '51-70', '71-90', '91-200']
        weights = [0.3, 0.3, 0.25, 0.2, 0.05]
        data_set.insert(2, 'age', random.choices(population, weights, k=n_agents))
    print(data_set)
    
    return data_set, ranking




### Check distribution of age in population
### Check distribution of age in sample
### Determine weights for each group
## Weighting data --> When data must be weighted, weight by as few variables as possible.
### As the number of weighting variables goes up, the greater the risk that the weighting of one variable will confuse or interact with the weighting of another variable.
### Most widely used tabulations systems and statistical packages use Iterative Proportional Fitting (or something similar)
### to weight survey data, a method popularized by the statistician Deming about 75 years ago.
# weight few variables to avoid conflict
# down weight (minimise groups) better than maximize groups

"""Use Raking [from ipfn import ipfn]
For public opinion surveys, the most prevalent method for weighting is iterative proportional fitting, more commonly referred to as raking.
With raking, a researcher chooses a set of variables where the population distribution is known,
and the procedure iteratively adjusts the weight for each case until the sample distribution aligns with the population for those variables.
For example, a researcher might specify that the sample should be 48% male and 52% female, and 40% with a high school education or less, 31% who have completed some college, and 29% college graduates.
The process will adjust the weights so that gender ratio for the weighted survey sample matches the desired population distribution.
Next, the weights are adjusted so that the education groups are in the correct proportion.
If the adjustment for education pushes the sex distribution out of alignment, then the weights are adjusted again so that men and women are represented in the desired proportion.
The process is repeated until the weighted distribution of all of the weighting variables matches their specified targets.
Raking is popular because it is relatively simple to implement, and it only requires knowing the marginal proportions for each variable used in weighting.
That is, it is possible to weight on sex, age, education, race and geographic region separately without having to first know the population proportion for every combination of characteristics
(e.g., the share that are male, 18- to 34-year-old, white college graduates living in the Midwest). Raking is the standard weighting method used by Pew Research Center and many other public pollsters.

In this study, the weighting variables were raked according to their marginal distributions, as well as by two-way cross-classifications for each pair of demographic variables 
(age, sex, race and ethnicity, education, and region).
    """
    
"""
1. Generate a dataset with n-dimensional group information (for an example we use Age and Party affiliation)
2. Get the information about cross-groups or just the numbers for the groups, can add as much or as little information we want!
3. Give back a randomly selected data set with a maximum possible size.    
"""
def generate_weighted_subset(survey, aggregates, dimensions, objective_size='max'):
    """[summary]

            example: aggregates = [[20,23], [23,23], [13,12, 12,12], xijp, xpjk]
            example: dimensions = [['party'], ['age'], ['age'], ['dma', 'size'], ['size', 'age']]
    Args:
        aggregates (list of lists): number of the objective's distributions 
        dimensions (list of lists): complements the dimensions for the list and specifies the aggregate's dimension
        objective_size (string/int):   maximum sample size (resample) but can't be bigger than the maximum
    """
    print(survey.groupby(['age', 'party_aff']).size())
    print(survey.nunique())
    print('----------------------------------')
    m = np.array(survey.groupby(['age', 'party_aff']).size())
    print('m is --------------------------', m)
    print('----------------------------------')
    m = np.reshape(m, (5,-1))           # reshape the matrix so that I got the right dimensions (1 dim for each attribute) #TODO make generic
    print(m)


    dimensions = [[0], [1]] #, [1], [2]]#, [0, 1], [1, 2]]
    m_copy = deepcopy(m)
    IPF = ipfn.ipfn(m_copy, aggregates, dimensions)
    m_copy = IPF.iteration()
    print(m_copy)
  

    # get maximum size for marix: get biggest factor ()
    print('results', m_copy/m)
    subset_matrix = (m_copy*1/np.max(m_copy/m))

    
    minimising_factor = objective_size/sum(sum(subset_matrix)) if objective_size !='max' else 1
    minimising_factor = minimising_factor if minimising_factor < 1 else 1
    print(f'{minimising_factor=}')

    print('maximum bootstrap size:', sum(sum(subset_matrix)))
    print(subset_matrix)
    subset_matrix = (subset_matrix *minimising_factor).round()      
    #print(m)
    
    # generate maximum bootstrap data randomly
    print('get first row', survey.groupby(['age', 'party_aff']).size().index.tolist())
    
    dataset_list = []
    for index_combination, n_needed in zip(survey.groupby(['age', 'party_aff']).size().index.tolist(), subset_matrix.flatten()):
        
        sub_survey_df = survey[survey['age']==index_combination[0]][survey['party_aff']==index_combination[1]]
        
        dataset_list = dataset_list + sub_survey_df.sample(int(n_needed)).values.tolist()
    
    ## final result list is the dataset_list
    print('Number of elements in the data list:', len(dataset_list))
    ## Define get a specific size of the data set.
    ## rescale the dataset:
    return dataset_list
    
#random_age_list = random.choices(['m', 'w'], k=10)
#print(random_age_list)
#print(generate_synth_attitude_surveys(split_up=[0.95, 0.05], age=True, n_agents=20)[0])
survey_df = generate_synth_attitude_surveys(split_up=[0.45, 0.55], age=True, n_agents=100)[0]

aggregates = [[20, 20, 10, 11, 2], [45, 55]]
dimensions = [[0], [1]]     #   [['age'],['party_aff']]

print(survey_df.nunique())

data_list = generate_weighted_subset(survey_df, aggregates, dimensions, objective_size=100)

#run_data_to_graph_to_community(data_list, n_items=10)




