import pickle
import time
import numpy as np
import random
import pandas as pd 
from scipy import stats
from sklearn.neighbors import KernelDensity
from ipfn import ipfn
from copy import deepcopy
from ANES_HierClus import rescaling_data
import ANES_com_detec_remodel_biggest_component
import networkx as nx
import matplotlib.pyplot as plt

#example
from sklearn.neighbors import KernelDensity
import networkx.algorithms.community as nx_comm

# 12/11/2021
def get_distributed_values(data='truncnorm',
                           n_size=100,
                           lower_limit=0.5,
                           upper_limit=7.5,
                           mu=4,
                           sigma=1,
                           n_elements = 2,
                           round_to_int=True
                           ):
    if isinstance(data, str):
        ### select distribution
        if 'truncnorm'==data:
            values = stats.truncnorm.rvs((lower_limit-mu)/sigma,(upper_limit-mu)/sigma,loc=mu,scale=sigma,size=n_size)
        elif 'uniform'==data:
            values = stats.uniform.rvs(size=n_size, loc=lower_limit, scale=upper_limit-lower_limit)
        elif 'n_elements'==data:
            n_elements = [int(round(el)) for el in stats.uniform.rvs(size=n_elements, loc=lower_limit, scale=upper_limit-lower_limit)]
            values = np.random.choice(a=n_elements, size=n_size, replace=True)
        else:
            print('select different distribution!')
        if round_to_int:
            values = [int(round(el)) for el in values]
    else:
        # getting distribution from data
        #### something like this:
        r = np.array(data).reshape(-1, 1)
        kd = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(r)
        values = np.random.choice([int(el+0.5) for el in kd.sample(10*n_size) if lower_limit < el < upper_limit], size=n_size, replace=True)
    
    return values



def create_simulated_data_set():
    """
    Create simulated data:
        - create x amout of groups
        - create x amount of questions
        - create x categories to generate participants
        - define a distribution by inserting data/using unified distribution/normal and construct participants/questions
        - add noise?
    """

    df = pd.DataFrame()
    
    # randomized construction
    # groupwise!
    group_conditions = {'Rep': {'n_size':200, 'n_categories':1, 'n_questions': 4, 'lower_limit':1, 'upper_limit': 7, 'mu': 2},
                        'Cen': {'n_size':200, 'n_categories':1, 'n_questions': 4, 'lower_limit':1, 'upper_limit': 7, 'mu': 4},
                        'Dem': {'n_size':200, 'n_categories':1, 'n_questions': 4, 'lower_limit':1, 'upper_limit': 7, 'mu': 6}}
    
    for group, condition in group_conditions.items():
        # generate questions
        #randomized?
        df_items = pd.DataFrame()
        for question in range(1, condition['n_questions']+1):
            data_generation = random.sample(['truncnorm', 'uniform', 'n_elements'], 1)[0]
            values = get_distributed_values(data=data_generation, n_size=condition['n_size'])
            df_items[f'Q_{question}'] = values
        
        # generate categories
        #randomized?
        df_categories = pd.DataFrame()
        # add group name
        
        #add categories
        for category in range(1, condition['n_categories']+1):
            data_generation = random.sample(['uniform', 'n_elements'], 1)[0]
            values = get_distributed_values(data=data_generation,
                                            n_size=condition['n_size'],
                                            n_elements=random.randint(condition['lower_limit'], condition['upper_limit']),
                                            lower_limit=condition['lower_limit'],
                                            upper_limit=condition['upper_limit'],
                                            mu=condition['mu']
                                            )
            df_categories[f'C_{category}'] = values
        
        df_categories.insert(0, 'party_aff', group)
        
        # 2 in 1 
        df_group = df_categories.join(df_items)

        # add to main df
        df = df.append(df_group).reset_index(drop=True)
    
    # add ID and return df
    df.insert(0, 'ID', range(1000, 1000 + len(df)))
    print(df.head(50))
    # descriptive statistics
    print(df.describe())
    print(df.groupby('party_aff').describe())
    return df

### select groups by similarity of questions:
# go through the groups and define the categories and maybe items that are required.
# random??
# requirements: category 1  , 2 and similarity in question: '1': [1,2,3,4] and questions: '1':
## using eval for the group selection process

#group 1:
# rest:


## groups after conditions??????
# groupnames at the moment only as number because of string comparison.
#example

def select_groups_by_conditions(df, conditions, groupsize):
    """
        conditions (Dictionary): {groupname: 'condition', ...}
        #groupsize (Dictionary): {groupname: groupsize, ...} commented
    """
    conditions ={1: '10 < df.Age < 20 & df[0] > 0', 2: '6 < df.Age < 20 & df[0] > 0'}
    #FOR GROUPSIZE group_size = {1: 40, 2: 40}
    # all participants are -1 group
    df['group'] = -1
    # group and conditions
    for group, condition in conditions.items():
        # minimum criteria is NOT a group member already = if they overlap than the first defined group is valid
        df.loc[pd.eval(f'({condition})' + '& df.group == -1'),'group'] = group
        # check if the group size is reachable:
        #FOR GROUPSIZE if len(df[df['group']==group]) < group_size[group]:
        #FOR GROUPSIZE    raise ValueError("condition is to strict, has to be reduced")


def generate_bias_group_selection(survey_df, groups, group_sizes):
    df = pd.DataFrame()
    for group, group_size in zip(groups, group_sizes):
        df_group = survey_df[survey_df['party_aff'] == group].sample(group_size, replace=replace)
        df = df.append(df_group).reset_index(drop=True)
    return df


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


def generate_weighted_subset(survey, aggregates, categories, objective_size='max', replace=False):
    """[summary]

            example: aggregates = [[20,23], [23,23], [13,12, 12,12], xijp, xpjk]
            example: categories = [['party'], ['age'], ['age'], ['dma', 'size'], ['size', 'age']]
    Args:
        aggregates (list of lists): number of the objective's distributions 
        categories (list of lists): complements the dimensions for the list and specifies the aggregate's dimension
        objective_size (string/int):   maximum sample size (resample) but can't be bigger than the maximum
    """
    # just to start
    print(survey.groupby(categories).size())
    print(survey.nunique())
    print('----------------------------------')
    m = np.array(survey.groupby(categories).size())
    print(m)
    print('----------------------------------')
    # generate tuple to reshape matrix --> number of different elements of dimension. for example age --> 4 classes --> 4 in tuple
    m_tuple = tuple([len(el) for el in aggregates])
    m = np.reshape(m, m_tuple)           # reshape the matrix so that I got the right dimensions (1 dim for each attribute) #TODO make generic
    print(m)


    dimensions = [[i] for i in range(len(aggregates))] #, [1], [2]]#, [0, 1], [1, 2]]
    m_copy = deepcopy(m)
    IPF = ipfn.ipfn(m_copy, aggregates, dimensions)
    m_copy = IPF.iteration()
    print(m_copy)
  

    # get maximum size for marix: get biggest factor ()
    print('results', m_copy/m)
    subset_matrix = (m_copy*1/np.max(m_copy/m))

    # get sum of matrix - dependend on the number of elements in subset_matrix
    if replace:
        minimising_factor = objective_size/np.sum(np.array(subset_matrix))
    else:
        minimising_factor = objective_size/np.sum(np.array(subset_matrix)) if objective_size !='max' else 1
        minimising_factor = minimising_factor if minimising_factor < 1 else 1
    print(f'{minimising_factor=}')

    print('Bootstrap size:', np.sum(np.array(subset_matrix)))
    print(subset_matrix)
    subset_matrix = (subset_matrix *minimising_factor).round()      
    #print(m)
    
    # generate maximum bootstrap data randomly
    print('get first row', survey.groupby(categories).size().index.tolist())
    
    dataset_list = []
    for index_combination, n_needed in zip(survey.groupby(categories).size().index.tolist(), subset_matrix.flatten()):
        
        # generate a sub sub sub sub .. survey to get a certain group || can be flexible in size
        sub_survey_df = survey
        if isinstance(index_combination, tuple):    # more than one category
            for position, category in enumerate(categories):
                sub_survey_df =sub_survey_df[sub_survey_df[category]==index_combination[position]]
        else: # only one category [problem solved: index_combination is not a tuple here!]
            sub_survey_df =sub_survey_df[sub_survey_df[categories[0]]==index_combination]
        
        # add a selected sample of df to the dataset_list
        dataset_list += sub_survey_df.sample(int(n_needed), replace=replace).values.tolist()
    
    ## final result list is the dataset_list
    print('Number of elements in the data list:', len(dataset_list))
    ## Define get a specific size of the data set.
    ## rescale the dataset:
    return dataset_list


def get_randomly_selected_subset(survey_df, objective_size=100, replace=False): ## bootstrap if with replacememt
    return survey_df.sample(objective_size, replace=replace).values.tolist()
    

def generate_synth_attitude_surveys(n_agents=100, n_questions=10, scale_steps=10, mu_max=5, number_ranks=5, n_comp=2,
                                        sd=0.5, split_up=[0.5, 0.5], additional_attributes={}):
    """ Generates overall popupaltion
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
    
    
    # Add age
    for column_name, pop_and_weights in additional_attributes.items():
        if len(pop_and_weights[0]) != len(pop_and_weights[1]):
            raise ValueError(column_name, ": population groups and weights musst be the same length!")
        population = pop_and_weights[0] #['18-30', '31-50', '51-70', '71-90', '91-200']
        weights = pop_and_weights[1] #[0.3, 0.3, 0.25, 0.2, 0.05]
        data_set.insert(2, column_name, random.choices(population, weights, k=n_agents))
    print(data_set)
    
    return data_set, ranking


def extract_multiple_communities(g, data_list):
    """ extract multiple communities from data list

    Args:
        g (networkx): graph with all the participants
        data_list (list): data in list

    Returns:
        dictionary: keys are the categories like e.g. 1 fop republican,...
    """
    # define as dict from list to find it easier
    participants_dict = {line[0]:line[1:] for line in data_list}

    # Get the Republican and the democrats of the sample
    communities = {}
    for i in zip(g.nodes()):
        
        # Check whether it is Republican or Democrat
        
        # from tuple to  int
        node = i[0]
        # generate new list if key is not in the community
        if not participants_dict.get(node)[0] in communities.keys():
            communities[participants_dict.get(node)[0]] =[]
        
        communities[participants_dict.get(node)[0]].append(node)
    return communities

def generate_synthic_population():
      #define population
    n_questions = 4
    
    survey_df = create_simulated_data_set()
    #change format
    total_data_list = survey_df.values.tolist()
    # define how many questions there are and rescale 
    total_rescaled_data_list = rescaling_data(total_data_list, scale_length_list=[7]*n_questions)    #[10] == number of response options i.e. 1-10
    
    total_gn_graph_dict, rescaled_features, threshold, edges_between_comp_renamed = ANES_com_detec_remodel_biggest_component.get_graph(
        total_data_list,
        total_rescaled_data_list,
        max_number_cluster=1,
        #minimum_size_for_two_comp=resplit
        )
        # threshold=7.0
    #community1, community2 = extract_two_communities(total_gn_graph_dict[1], total_data_list)    # give in graph and data_list
    return survey_df, total_gn_graph_dict[1]
    print('number of same nodes', len([edge for edge in total_gn_graph_dict[1].edges(data=True) if edge[2]['weight'] == n_questions]))

if __name__ == "__main__":
    ## main program
    
    if False:
        survey_df, population_network = generate_synthic_population()
        
        #-SAVING
        #save and load network
        with open('population.pkl', 'wb') as f:
            pickle.dump(population_network, f)
        
        #-SAVING data
        with open('population_df.pkl', 'wb') as f:
            pickle.dump(survey_df, f)
    #---------------------------------------------------------------------
    n_questions = 4

    #load network
    with open('population.pkl', 'rb') as f:
        population_network = pickle.load(f)
    
    #load data
    with open('population_df.pkl', 'rb') as f:
        survey_df = pickle.load(f)
    
    print(survey_df)
    
    # check groups
    group_graphs = {}
    for group in ['Rep', 'Cen','Dem']:
        print('Group', group)
        group_members = survey_df[survey_df['party_aff']==group]['ID'].values.tolist()
        group_graphs[group] = population_network.subgraph(group_members)
        print('number of same nodes', len([edge for edge in graph_group.edges(data=True) if edge[2]['weight'] == n_questions]))

    # -----------------------end:population and network defined


    print('Start: bootstrap')
    # get bootstrap-samples from network
    objective_size=750
    replace=True
    df_bootstrap = survey_df.sample(objective_size, replace=replace) #with replacement
    
    #change format
    bootstrap_data_list = df_bootstrap.values.tolist()
    # define how many questions there are and rescale 
    bootstrap_rescaled_data_list = rescaling_data(bootstrap_data_list, scale_length_list=[7]*n_questions)    #[10] == number of response options i.e. 1-10
    
    bootstrap_gn_graph_dict, rescaled_features, threshold, edges_between_comp_renamed = ANES_com_detec_remodel_biggest_component.get_graph(
        bootstrap_data_list,
        bootstrap_rescaled_data_list,
        max_number_cluster=1,
        #minimum_size_for_two_comp=resplit
        )
        # threshold=7.0
    #community1, community2 = extract_two_communities(total_gn_graph_dict[1], total_data_list)    # give in graph and data_list

    print('number of same nodes', len([edge for edge in bootstrap_gn_graph_dict[1].edges(data=True) if edge[2]['weight'] == n_questions]))
    
    # check groups
    for group in ['Rep', 'Cen','Dem']:
        print('Group', group)
        group_members = df_bootstrap[df_bootstrap['party_aff']==group]['ID'].values.tolist()
        graph_group = bootstrap_gn_graph_dict[1].subgraph(group_members)
        print('number of same nodes', len([edge for edge in graph_group.edges(data=True) if edge[2]['weight'] == n_questions]))
    
    # get by condition: df.loc[pd.eval(f'({condition})')]     define group and condition then choose with replacement



def example_mean():
    #compare bootstraped network and original network
    #example: mean
    Q1_mean_original = survey_df['Q_1'].mean()
    print('Q1_mean_original', Q1_mean_original)
    bootstrapped_Q1_mean = []
    for i in range(10):
        df_bootstrap = survey_df.sample(objective_size, replace=replace)
        bootstrapped_Q1_mean.append(df_bootstrap['Q_1'].mean())
        if i%500 == 0:
            print(np.mean(bootstrapped_Q1_mean))
    print('end: ', np.mean(bootstrapped_Q1_mean), 'distance to original', Q1_mean_original-np.mean(bootstrapped_Q1_mean))
    
    survey_df.duplicated(subset=['Q_1','Q_2', 'Q_3', 'Q_4']).sum()
    
    """
    #example: same rows in the dataset
    same_rows_original = survey_df.duplicated(subset=['Q_1','Q_2', 'Q_3', 'Q_4']).sum()
    print('Q1_mean_original', same_rows_original)
    bootstrapped_same_rows = []
    for i in range(10000):
        df_bootstrap = survey_df.sample(objective_size, replace=replace)
        bootstrapped_same_rows.append(df_bootstrap.duplicated(subset=['Q_1','Q_2', 'Q_3', 'Q_4']).sum())
        if i%500 == 0:
            print(np.mean(bootstrapped_same_rows))
    print('end: ', np.mean(bootstrapped_same_rows), 'distance to original', same_rows_original-np.mean(bootstrapped_same_rows))    
    """
    
    #generate bias --> by group
    groups = ['Rep', 'Cen', 'Dem']
    survey_df['Q_1'].plot.density(bw_method=0.3, label='Q_1_original')
    #survey_df['Q_2'].plot.hist(bins=7, alpha=0.1)
    #survey_df['Q_3'].plot.hist(bins=7, alpha=0.1)
    #survey_df['Q_4'].plot.hist(bins=7, alpha=0.1)

    plt.savefig('test_file.png')
    df_bootstrap = generate_bias_group_selection(survey_df, groups, group_sizes=[650, 50, 50])

    #example: mean
    print('example: BIAS selection groups')
    Q1_mean_original = survey_df['Q_1'].mean()
    print('Q1_mean_original', Q1_mean_original)
    bootstrapped_Q1_mean = []
    boostrap_runs = 100
    df_bootstrap_all = pd.DataFrame()
    
    for i in range(boostrap_runs):
        df_bootstrap = generate_bias_group_selection(survey_df, groups, group_sizes=[650, 50, 50])
       
        bootstrapped_Q1_mean.append(df_bootstrap['Q_1'].mean())
        if i%500 == 0:
            print(np.mean(bootstrapped_Q1_mean))
        df_bootstrap_all = df_bootstrap_all.append(df_bootstrap).reset_index(drop=True)
    print('end: ', np.mean(bootstrapped_Q1_mean), 'distance to original', Q1_mean_original-np.mean(bootstrapped_Q1_mean))
    df_bootstrap_all['Q_1'].plot.density(bw_method=0.3, label='Q_1_bootstrapped')
    plt.legend()
    plt.savefig('hello_bar.png')
    
    
    ##Example: Path length in network weighted vs random vs total
    # Degree measure?
    # Betweenness measure?
    # Closeness measure?
    # Eigenvector measure?
    # Pagerank?
    # HITS?                 all from https://arxiv.org/pdf/1703.03741.pdf
















"""
########### --------------------------------------

## main program
#1 Generate a population for the ground truth
# Spezifications: 10000 people, divided in two groups 45 % to 55 % (variable)
#       add:    hurling-fan: 'Cork', 'Limerick', 'Kilkenny'
#               age-groups: '10-20', '20-30', '30-67', '68-111'
#               salary: '0-1000', '1001-3000', '3001 - 7000', '7001-1000000'    --> should be provided as a specific salary like age
#               job: 'industry', 'university', 'other'
n_people = 300
n_questions = 10

# give in CATEGORY: [[class, class2, class3, class4, class5, ...], [probability, probability2, probability3, ...]]
additional_attributes = {
    'hurling-fan': [['Cork', 'Limerick', 'Kilkenny'], [0.4, 0.4, 0.2]],
    'age': [[ '10-20', '20-30', '30-67', '68-111'], [0.4, 0.4, 0.1, 0.1]],
    'salary': [['0-1000', '1001-3000', '3001 - 7000', '7001-1000000'], [0.5, 0.3, 0.1, 0.1]],
    'boolean': [['positive', 'negative'], [0.3, 0.7]]
}

survey_df = generate_synth_attitude_surveys(split_up=[0.30, 0.70], additional_attributes=additional_attributes, n_agents=n_people, n_questions=n_questions)[0]

# SELECT FOR RESAMPLING

#### 2.0 Generate population network --------------------------------------------------------------------
resplit = False
total_data_list = survey_df.values.tolist()
total_rescaled_data_list = rescaling_data(total_data_list, scale_length_list=[10]*n_questions)    #[10] == number of response options i.e. 1-10
total_gn_graph_dict, rescaled_features, threshold, edges_between_comp_renamed = ANES_com_detec_remodel_biggest_component.get_graph(
    total_data_list,
    total_rescaled_data_list,
    max_number_cluster=1,
    minimum_size_for_two_comp=resplit,
    # threshold=7.0
)

# extract the communities so that I can calculate the entropy in a second step
# 1
community1, community2 = extract_two_communities(total_gn_graph_dict[1], total_data_list)    # give in graph and data_list
print('TOTAL: Degree assortatitvity coefficient', nx.degree_assortativity_coefficient(total_gn_graph_dict[1]))
# get shortest path length for 2 communities
community1_subgraph = total_gn_graph_dict[1].subgraph(set(community1))
total_community1_shortest_path = np.mean(compute_shortest_path_length(community1_subgraph))
community2_subgraph = total_gn_graph_dict[1].subgraph(set(community2))
total_community2_shortest_path = np.mean(compute_shortest_path_length(community2_subgraph))

total_EB_score = EB_polarisation_score(total_gn_graph_dict[1], community1, community2)

#### END Generate population network --------------------------------------------------------------------


##Example: Path length in network weighted vs random vs total

# Degree measure?
# Betweenness measure?
# Closeness measure?
# Eigenvector measure?
# Pagerank?
# HITS?                 all from https://arxiv.org/pdf/1703.03741.pdf


########## SAMPLING Random and correctly weighted
# here to put in the proportions needed for sample
aggregates = [[40, 40, 10, 10], [30, 70]]#, [0.4, 0.4, 0.2, 0.3]]
# put in dimensions (do I need that?), better give in names
categories = ['age', 'party_aff']#, 'salary'] #[[0], [1], [2]]     #   [['age'],['party_aff']] 
# get weighted survey from data
weighted_data_list = generate_weighted_subset(survey_df, aggregates, categories, objective_size=200, replace=False)
weighted_survey_df = pd.DataFrame(weighted_data_list, columns=survey_df.columns)
#### 2.1 Generate network from weighted resample:
# Resplit the network if minimum group size is not reached?
resplit = False

rescaled_weighted_data_list = rescaling_data(weighted_data_list, scale_length_list=[10]*n_questions)    #[10] == number of response options i.e. 1-10
weighted_gn_graph_dict, rescaled_features, threshold, edges_between_comp_renamed = ANES_com_detec_remodel_biggest_component.get_graph(
    weighted_data_list,
    rescaled_weighted_data_list,
    max_number_cluster=1,
    minimum_size_for_two_comp=resplit,
    # threshold=7.0
)

# get polarization score for the weighted data 
community1, community2 = extract_two_communities(weighted_gn_graph_dict[1], weighted_data_list)    # give in graph and data_list
print('WEIGHTED Degree assortatitvity coefficient', nx.degree_assortativity_coefficient(weighted_gn_graph_dict[1]))
# get shortest path length for 2 communities
community1_subgraph = weighted_gn_graph_dict[1].subgraph(set(community1))
weightedsample_community1_shortest_path = np.mean(compute_shortest_path_length(community1_subgraph))
community2_subgraph = weighted_gn_graph_dict[1].subgraph(set(community2))
weightedsample_community2_shortest_path = np.mean(compute_shortest_path_length(community2_subgraph))

#total
shortest_path_weighted_network_sample = np.mean(compute_shortest_path_length(weighted_gn_graph_dict[1]))
EB_polarisation_score_weighted_sample = EB_polarisation_score(weighted_gn_graph_dict[1], community1, community2)
                                                            # SHOULD I SET THE SAME SIZE AS WEIGHTED?
size = len(rescaled_weighted_data_list)
print(size)
#### 2.1 Generate network randomly
# get subset_matrix
random_data_list = get_randomly_selected_subset(survey_df, size, False)
random_survey_df = pd.DataFrame(random_data_list, columns=survey_df.columns)
rescaled_random_data_list = rescaling_data(random_data_list, scale_length_list=[10]*n_questions)    #[10] == number of response options i.e. 1-10
random_gn_graph_dict, rescaled_features, threshold, edges_between_comp_renamed = ANES_com_detec_remodel_biggest_component.get_graph(
    random_data_list,
    rescaled_random_data_list,
    max_number_cluster=1,
    minimum_size_for_two_comp=resplit,
    # threshold=7.0
)
# get polarization score for the weighted data 
community1, community2 = extract_two_communities(random_gn_graph_dict[1], random_data_list)    # give in graph and data_list

print('Random Degree assortatitvity coefficient', nx.degree_assortativity_coefficient(random_gn_graph_dict[1]))
# get shortest path length for 2 communities
community1_subgraph = random_gn_graph_dict[1].subgraph(set(community1))
randomsample_community1_shortest_path = np.mean(compute_shortest_path_length(community1_subgraph))
community2_subgraph = random_gn_graph_dict[1].subgraph(set(community2))
randomsample_community2_shortest_path = np.mean(compute_shortest_path_length(community2_subgraph))
#total
shortest_path_random_network_sample = np.mean(compute_shortest_path_length(random_gn_graph_dict[1]))

EB_polarisation_score_weighted_sample = EB_polarisation_score(random_gn_graph_dict[1], community1, community2)
#
#
#
# ReSampling from the sample

# shortestpath length total
shortest_path_weighted_network = []
shortest_path_random_network = []

# EB score list
weighted_EB_score = []
random_EB_score = []
#
# communities shortest path length
weighted_resample_community1_shortest_path = []
weighted_resample_community2_shortest_path = []
random_resample_community1_shortest_path = []
random_resample_community2_shortest_path = []
#
## RESAMPLING FROM THE SAMPLING -- sub sub sets
for _ in range(100):
    ## resampling from the weighted sample -- Do I need the weighting here?
    # here to put in the proportions needed for the bootstrap!
    aggregates = [[40, 40, 10, 10], [30, 70]]#, [0.4, 0.4, 0.2, 0.3]]

    # put in dimensions (do I need that?), better give in names
    categories = ['age', 'party_aff']#, 'salary'] #[[0], [1], [2]]     #   [['age'],['party_aff']] 
    # get weighted survey from data
    weighted_data_list = generate_weighted_subset(weighted_survey_df, aggregates, categories, objective_size=100, replace=True)

    #### 2.1 Generate network from weighted resample:
    # Resplit the network if minimum group size is not reached?
    resplit = False

    rescaled_weighted_data_list = rescaling_data(weighted_data_list, scale_length_list=[10]*n_questions)    #[10] == number of response options i.e. 1-10
    weighted_gn_graph_dict, rescaled_features, threshold, edges_between_comp_renamed = ANES_com_detec_remodel_biggest_component.get_graph(
        weighted_data_list,
        rescaled_weighted_data_list,
        max_number_cluster=1,
        minimum_size_for_two_comp=resplit,
        # threshold=7.0
    )
    
    # get polarization score for the weighted data 
    community1, community2 = extract_two_communities(weighted_gn_graph_dict[1], weighted_data_list)    # give in graph and data_list
    
    # separated communities
    # get shortest path length for 2 communities
    community1_subgraph = weighted_gn_graph_dict[1].subgraph(set(community1))
    weighted_resample_community1_shortest_path.append(np.mean(compute_shortest_path_length(community1_subgraph)))
    community2_subgraph = weighted_gn_graph_dict[1].subgraph(set(community2))
    weighted_resample_community2_shortest_path.append(np.mean(compute_shortest_path_length(community2_subgraph)))
    
    weighted_EB_score.append(EB_polarisation_score(weighted_gn_graph_dict[1], community1, community2))
                                                                # SHOULD I SET THE SAME SIZE AS WEIGHTED?
    size = len(rescaled_weighted_data_list)
    print(size)
    #### 2.1 Generate network randomly
    # get subset_matrix
    random_data_list = get_randomly_selected_subset(random_survey_df, size, False)
    rescaled_random_data_list = rescaling_data(random_data_list, scale_length_list=[10]*n_questions)    #[10] == number of response options i.e. 1-10
    random_gn_graph_dict, rescaled_features, threshold, edges_between_comp_renamed = ANES_com_detec_remodel_biggest_component.get_graph(
        random_data_list,
        rescaled_random_data_list,
        max_number_cluster=1,
        minimum_size_for_two_comp=resplit,
        # threshold=7.0
    )
    # get polarization score for the weighted data 
    community1, community2 = extract_two_communities(random_gn_graph_dict[1], random_data_list)    # give in graph and data_list
    
    # get shortest path length for 2 communities
    community1_subgraph = random_gn_graph_dict[1].subgraph(set(community1))
    random_resample_community1_shortest_path.append(np.mean(compute_shortest_path_length(community1_subgraph)))
    community2_subgraph = random_gn_graph_dict[1].subgraph(set(community2))
    random_resample_community2_shortest_path.append(np.mean(compute_shortest_path_length(community2_subgraph)))
    
    random_EB_score.append(EB_polarisation_score(random_gn_graph_dict[1], community1, community2))
    
    shortest_path_weighted_network.append(np.mean(compute_shortest_path_length(weighted_gn_graph_dict[1])))
    shortest_path_random_network.append(np.mean(compute_shortest_path_length(random_gn_graph_dict[1])))

print('------------------------------------------------SHORTEST AVG PATH [total]------------------------------------------------------')
print('SHORTEST AVG PATH', np.mean(compute_shortest_path_length(total_gn_graph_dict[1])))
print('Samples', f'{shortest_path_weighted_network_sample=} {shortest_path_random_network_sample=}')
print('weighted ReSample', np.mean(shortest_path_weighted_network), '||', 'random ReSample', np.mean(shortest_path_random_network))
print('------------------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------SHORTEST AVG PATH [communities]------------------------------------------------------')
print(f'{total_community1_shortest_path=}')
print(f'{total_community2_shortest_path=}')
print('SAMPLES:')
print(f'{weightedsample_community1_shortest_path=}')
print(f'{weightedsample_community2_shortest_path=}')
print(f'{randomsample_community1_shortest_path=}')
print(f'{randomsample_community2_shortest_path=}')
print('Resamples:')
for i, name in zip([weighted_resample_community1_shortest_path, weighted_resample_community2_shortest_path, random_resample_community1_shortest_path, random_resample_community2_shortest_path],
             ['weighted_resample_community1_shortest_path', 'weighted_resample_community2_shortest_path', 'random_resample_community1_shortest_path', 'random_resample_community2_shortest_path']
             ):
    print(name, np.mean(i))
print('------------------------------------------------------------------------------------------------------')
print('EB_polarisation_score')
print('random:', np.mean(random_EB_score))
print('weighted:', np.mean(weighted_EB_score))
print('total', total_EB_score)
"""