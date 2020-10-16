import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import os
import itertools
import time

path = '/~~~'

expansion_plan = pd.read_csv(path + '/expansion_plan_1.csv', header= None)
candidate_plan1_indices = expansion_plan.values[np.where(expansion_plan.values[:,1] == 1.),0]

lines = pd.read_csv(os.path.join(path, 'lines.csv'))
possible_lines = lines.values[:,[1,2,3,10]]
existing_lines = possible_lines[np.where(possible_lines[:,2] ==1.),:]
existing_lines_indices = lines.values[np.where(possible_lines[:,2] == 1.),0].astype(np.int32)


expansion_plan1_indices = np.concatenate((existing_lines_indices, candidate_plan1_indices), axis = 1) # candidate_plan1_indices[:,0].reshape(1,1)
expansion_plan1_edges = lines.values[np.ix_((expansion_plan1_indices-1).reshape(expansion_plan1_indices.shape[1],),[1,2,10])]
expansion_plan1_edges[:,0].astype(np.int32)
expansion_plan1_edges[:,1].astype(np.int32)

input_edges = [(int(u), int(v),w) for u,v,w in expansion_plan1_edges]
expansion_plan1_graph = nx.Graph()
expansion_plan1_graph.add_weighted_edges_from(input_edges)
nx.draw_networkx(expansion_plan1_graph, with_labels = True)
#print(expansion_plan1_graph.edges(data = True))

# between load and substation #
expansion_graph = expansion_plan1_graph
uniqueness_score_col = np.zeros(shape=(50,4), dtype= np.float32)

# start time #
start_time = time.time()

for sub_id in range(4):
    print(sub_id)
    uniqueness_score_mat = np.zeros(shape=(50, ), dtype=np.float32)
    for target_load in np.arange(1, 51):
        tmp_load_tmp_sub = nx.all_simple_paths(expansion_graph, source=target_load, target=(51+sub_id))
        paths_tmp_load_tmp_sub = list(tmp_load_tmp_sub)
        pairwise_paths = [list(path) for path in
                          map(nx.utils.pairwise,
                              paths_tmp_load_tmp_sub)]  # get pairwise path between "from" node and "to" node

        path_len = np.zeros((len(pairwise_paths), 1), dtype=np.float32)
        for i in range(len(pairwise_paths)):
            tmp_path = pairwise_paths[i]
            tmp_path_weighted_sum = 0.
            for j in range(len(tmp_path)):
                tmp_path_weighted_sum = tmp_path_weighted_sum + \
                                        expansion_graph[tmp_path[j][0]][tmp_path[j][1]]["weight"]
            path_len[i] = tmp_path_weighted_sum

        path_len_order = np.argsort(path_len, axis=0)
        ordered_paths_tmp_load_tmp_sub = np.array(paths_tmp_load_tmp_sub)[path_len_order]
        ordered_lists_tmp_load_tmp_sub = list(
            itertools.chain.from_iterable(ordered_paths_tmp_load_tmp_sub.tolist()))  # sorted lists

        # step 1 #
        uniqueness_score_vec = np.zeros((len(paths_tmp_load_tmp_sub),), dtype=np.float32)
        tmp_union = []
        for uu in range(len(paths_tmp_load_tmp_sub)):
            if uu == 0:
                uniqueness_score_vec[uu] = 1.
                tmp_union = tmp_union + ordered_lists_tmp_load_tmp_sub[uu]
            else:
                # only load & read previous information rather than read whole lists
                tmp_intersect = np.intersect1d(ordered_paths_tmp_load_tmp_sub[uu][0], tmp_union)
                tmp_union = tmp_union + ordered_lists_tmp_load_tmp_sub[uu]
                tmp_union = sorted([*{*tmp_union}])
                # end #
                intersect_len = len(tmp_intersect)
                if (target_load in tmp_intersect) & ((51 + sub_id) in tmp_intersect):
                    intersect_len = intersect_len - 2
                tmp_uniqueness_score = 1 - intersect_len / (len(ordered_paths_tmp_load_tmp_sub[uu][0]) - 2)
                uniqueness_score_vec[uu] = tmp_uniqueness_score

        # step 2 #
        path_len_vec = np.zeros((len(paths_tmp_load_tmp_sub),), dtype=np.float32)
        for vv in range(len(paths_tmp_load_tmp_sub)):
            path_len_vec[vv] = len(ordered_paths_tmp_load_tmp_sub[vv][0]) - 1  # i.e., count the steps in path

        uniqueness_score_mat[target_load - 1] = np.sum(uniqueness_score_vec / path_len_vec)
    uniqueness_score_col[:,sub_id] = uniqueness_score_mat

print("--- %s seconds ---" % (time.time() - start_time))
print(uniqueness_score_col)




#print(uniqueness_score_mat)
#print(uniqueness_score_mat.sum()) # sum uniqueness score for all loads (only to one substation)
