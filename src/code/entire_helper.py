"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
MFRWR
- A comparative study of matrix factorization and
  random walk with restart in recommender system

Authors
- Haekyu Park (hkpark627@snu.ac.kr)
- Jinhong Jung (jinhongjung@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
  from Data Mining Lab., Seoul National University
  (https://datalab.snu.ac.kr)

File
- entire_helper.py
  : includes helper functions for both matrix factorization
    and random walk with restart methods.


This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import packages and define frequently used variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
header = ['u', 'i', 'r']
numfolds = 5


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Parse config file
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Read config file and parse it
def config_reader(args):
	config_file = args.config_path
	f = open(config_file)
	while True:
		line = f.readline()
		if line == '':
			break
		pair = line.split(';')
		if pair[0] == 'dataset':
			args.dataset = pair[1].split('\n')[0]
		if pair[0] == 'method':
			args.method = pair[1].split('\n')[0]			
		if pair[0] == 'data_path':
			args.data_path = pair[1].split('\n')[0]
		if pair[0] == 'input_path':
			args.input_path = pair[1].split('\n')[0]
		if pair[0] == 'result_path':
			args.result_path = pair[1].split('\n')[0]
		if pair[0] == 'side_paths':
			lst = pair[1].split('\n')[0][1:-1]
			lst = lst.split('\'')
			side_paths = list()
			for i, e in enumerate(lst):
				if i % 2 == 1:
					side_paths.append(e)
			args.side_paths = side_paths
		if pair[0] == 'entity_types':
			lst = pair[1].split('\n')[0][1:-1]
			lst = extract_lst(lst)
			args.entity_types = lst
		if pair[0] == 'config':
			boo = pair[1].split('\n')[0]
			if boo == 'True' or boo == 'true':
				args.config = True
			if boo == 'False' or boo == 'false':
				args.config = False
		if pair[0] == 'config_path':
			args.config_path = pair[1].split('\n')[0]
		if pair[0] == 'is_implicit':
			boo = pair[1].split('\n')[0]
			if boo == 'True' or boo == 'true':
				args.is_implicit = True
			if boo == 'False' or boo == 'false':
				args.is_implicit = False
		if pair[0] == 'alpha':
			args.alpha = float(pair[1].split('\n')[0])
		if pair[0] == 'is_social':
			boo = pair[1].split('\n')[0]
			if boo == 'True' or boo == 'true':
				args.is_social = True
			if boo == 'False' or boo == 'false':
				args.is_social = False
		if pair[0] == 'lr':
			args.lr = float(pair[1].split('\n')[0])
		if pair[0] == 'lamb':
			args.lamb = float(pair[1].split('\n')[0])
		if pair[0] == 'dim':
			args.dim = int(pair[1].split('\n')[0])
		if pair[0] == 'is_sample':
			boo = pair[1].split('\n')[0]
			if boo == 'True' or boo == 'true':
				args.is_sample = True
			if boo == 'False' or boo == 'false':
				args.is_sample = False
		if pair[0] == 'num_seed':
			args.num_seed = int(pair[1].split('\n')[0])
		if pair[0] == 'c':
			args.c = float(pair[1].split('\n')[0])
		if pair[0] == 'beta':
			args.beta = float(pair[1].split('\n')[0])
		if pair[0] == 'gamma':
			args.gamma = float(pair[1].split('\n')[0])
		if pair[0] == 'delta':
			args.delta = float(pair[1].split('\n')[0])

	f.close()

# Parse entity_types argument
def extract_lst(inputstr):
	lst = list()
	word_start = False
	smaller_lst = list()
	word = ''
	for s in inputstr:
		if s == '[':
			smaller_lst = list()
			continue
		elif s == ']':
			lst.append(smaller_lst)
			continue
		else:
			if s == '\'':
				if word_start:
					smaller_lst.append(word)
					word = ''
					word_start = False
				else:
					word_start = True
			elif s == ' ' or s == ',':
				continue
			else:
				word += s
	return lst


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Input related functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Split rating data into 5 folds ramdomly
def split_5_folds(args):
	K = 5
	names = ['user_id', 'item_id', 'rating']
	df = pd.read_csv(args.input_path + 'rating.tsv', sep='\t', names=names)
	ratings = coo_matrix((df.rating, (df.user_id, df.item_id)))
	users = np.unique(ratings.row)
	ratings = ratings.tocsr()

	rows = list()
	cols = list()
	vals = list()
	nonzeros = list()

	for k in range(K):
		size_of_bucket = int(ratings.nnz / K)
		if k == K - 1:
			size_of_bucket += ratings.nnz % K
		rows.append(np.zeros(size_of_bucket))
		cols.append(np.zeros(size_of_bucket))
		vals.append(np.zeros(size_of_bucket))
		nonzeros.append(0)

	for i, user in enumerate(users):
		items = ratings[user, :].indices
		rating_vals = ratings[user, :].data
		index_list = [i for i in range(K)] * int(len(items) / float(K) + 1)
		np.random.shuffle(index_list)
		index_list = np.array(index_list)

		for k in range(K):
			k_index_list = (index_list[:len(items)] == k)
			from_ind = nonzeros[k]
			to_ind = nonzeros[k] + sum(k_index_list)

			if to_ind >= len(rows[k]):
				rows[k] = np.append(rows[k], np.zeros(size_of_bucket))
				cols[k] = np.append(cols[k], np.zeros(size_of_bucket))
				vals[k] = np.append(vals[k], np.zeros(size_of_bucket))
				k_index_list = (index_list[:len(items)] == k)

			rows[k][from_ind:to_ind] = [user] * sum(k_index_list)
			cols[k][from_ind:to_ind] = items[k_index_list]
			vals[k][from_ind:to_ind] = rating_vals[k_index_list]
			nonzeros[k] += sum(k_index_list)

	for k, (row, col, val, nonzero) in enumerate(zip(rows, cols, vals, nonzeros)):
		bucket_df = pd.DataFrame({'user': row[:nonzero], 'item': col[:nonzero], 'rating': val[:nonzero]}, columns=['user', 'item', 'rating'])
		bucket_df.to_csv(args.input_path + 'b%d.csv' % k, sep='\t', header=False, index=False)


# Read rating and return train and test set
def read_rating(input_path, fold):
	train = pd.DataFrame()
	test = pd.DataFrame()
	for i in range(numfolds):
		input_rating = input_path + 'b%d.csv' % i
		b = pd.read_csv(input_rating, sep='\t', names=header)
		if i == fold:
			train = pd.concat([train, b])
		else:
			test = b
	return train, test


# Make a dictionary including test set data
def get_test_table(test):
	# Input: 
	#  - test set of three-columned data (u \t i \t r)
	# Output
	#  - dictionary of users' rating pattern
	#    ex) test_table[u] = [[i1, r1], [i2, r2], [i3, r3], ...]
	test_table = dict()
	for u, i, r in zip(test.u, test.i, test.r):
		u = int(u)
		i = int(i)
		if not (u in test_table):
			test_table[u] = list()
		test_table[u].append([i, r])
	return test_table



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Mapping entities' id and their sequence
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Map entities' id and their sequence
def map_entity(dat, e2th, th2e, num_specific, num_entity):
	# Input: 
	#  - e2th: a dictionary whose key is an entity's id 
	#          and value is its sequence
	#  - th2e: a dictionary whose key is an entity's sequence
	#          and value is its id
	#  - num_entity: the total number of entity
	#  - num_specific: the total number of members in 
	#                  the entity's class. (e.g, # users)
	# Output: 
	#  - e2th: updated e2th by adding information in dat
	#  - th2e: updated th2e by adding information in dat
	#  - num_entity: updated num_entity by adding information in dat
	#  - num_specific: updated num_specific by adding information in dat
	for e in dat:
		e = int(e)
		if not(e in e2th):
			e2th[e] = num_entity
			th2e[num_entity] = e
			num_specific += 1
			num_entity += 1
	return e2th, th2e, num_entity, num_specific


# Map entities' id and their sequence with side information
def map_entity_with_side(train, side_bucket, entity_types, e2th, th2e, num_e, num_u, num_i):
	# Input: 
	#  - train: train rating data
	#  - side_bucket: list of side information
	#  - entity_types: list of entity types for each side information
	#  - e2th: a dictionary whose key is an entity's id 
	#          and value is its sequence
	#  - th2e: a dictionary whose key is an entity's sequence
	#          and value is its id
	#  - num_e: number of entities (# users + # items)
	#  - num_u: number of users
	#  - num_i: number of items
	# Output: 
	#  - e2th: updated e2th by adding information in train data
	#  - th2e: updated th2e by adding information in train data
	#  - num_e: updated num_e by adding information in train data
	#  - num_u: updated num_u by adding information in train data
	#  - num_i: updated num_i by adding information in train data
	# Map users first
	e2th, th2e, num_e, num_u = map_entity(train.u, dict(), dict(), 0, 0)
	for sideth in range(len(entity_types)):
		first_entity_type = entity_types[sideth][0]
		second_entity_type = entity_types[sideth][1]
		if first_entity_type == 'u':
			e2th, th2e, num_e, num_u = map_entity(side_bucket[sideth].e1, e2th, th2e, num_u, num_e)
		if second_entity_type == 'u':
			e2th, th2e, num_e, num_u = map_entity(side_bucket[sideth].e2, e2th, th2e, num_u, num_e)
	# Map items
	e2th, th2e, num_e, num_i = map_entity(train.i, e2th, th2e, 0, num_e)
	for sideth in range(len(entity_types)):
		first_entity_type = entity_types[sideth][0]
		second_entity_type = entity_types[sideth][1]
		if first_entity_type == 'i':
			e2th, th2e, num_e, num_i = map_entity(side_bucket[sideth].e1, e2th, th2e, num_i, num_e)
		if second_entity_type == 'i':
			e2th, th2e, num_e, num_i = map_entity(side_bucket[sideth].e2, e2th, th2e, num_i, num_e)
	# Map others
	for sideth in range(len(entity_types)):
		first_entity_type = entity_types[sideth][0]
		second_entity_type = entity_types[sideth][1]
		if first_entity_type == 's':
			e2th, th2e, num_e, _ = map_entity(side_bucket[sideth].e1, e2th, th2e, 0, num_e)
		if second_entity_type == 's':
			e2th, th2e, num_e, _ = map_entity(side_bucket[sideth].e2, e2th, th2e, 0, num_e)

	return e2th, th2e, num_e, num_u, num_i


# Map entities' id and their sequence with side information
#  by separately considering users and items
def map_ui_with_side(train, side_bucket, entity_types):
	# Input: 
	#  - train: train rating data
	#  - side_bucket: list of side information
	#  - entity_types: list of entity types for each side information
	# Output: 
	#  - u2th: a dictionary whose key is users' id and value is their sequence
	#  - th2u: a dictionary whose key is users' sequence and value is their id
	#  - i2th: a dictionary whose key is users' id and value is their sequence
	#  - th2i: a dictionary whose key is users' sequence and value is their id
	#  - num_u: number of users
	#  - num_i: number of items
	# Map users
	u2th, th2u, _, num_u = map_entity(train.u, dict(), dict(), 0, 0)
	for sideth in range(len(entity_types)):
		first_entity_type = entity_types[sideth][0]
		second_entity_type = entity_types[sideth][1]
		if first_entity_type == 'u':
			u2th, th2u, _, num_u = map_entity(side_bucket[sideth].e1, u2th, th2u, num_u, 0)
		if second_entity_type == 'u':
			u2th, th2u, _, num_u = map_entity(side_bucket[sideth].e2, u2th, th2u, num_u, 0)
	# Map items
	i2th, th2i, _, num_i = map_entity(train.i, dict(), dict(), 0, 0)
	for sideth in range(len(entity_types)):
		first_entity_type = entity_types[sideth][0]
		second_entity_type = entity_types[sideth][1]
		if first_entity_type == 'i':
			e2th, th2e, _, num_i = map_entity(side_bucket[sideth].e1, i2th, th2i, num_i, 0)
		if second_entity_type == 'i':
			e2th, th2e, _, num_i = map_entity(side_bucket[sideth].e2, i2th, th2i, num_i, 0)
	return u2th, th2u, i2th, th2i, num_u, num_i


# Map attributes' id and their sequence
def map_attributes(side_bucket, entity_types):
	# Input: 
	#  - side_bucket: list of side information
	#  - entity_types: list of entity types for each side information
	# Output: 
	#  - sideth2usth: a dictionary whose key is user similarity attributes' id 
	#                  and value is their sequence
	#  - usth2sideth: a dictionary whose key is user similarity attributes' sequence 
	#                  and value is their id
	#  - sideth2usth: a dictionary whose key is item similarity attributes' id 
	#                  and value is their sequence
	#  - usth2sideth: a dictionary whose key is item similarity attributes' sequence 
	#                  and value is their id
	#  - num_ustypes: number of user similarity attributes
	#  - num_istypes: number of item similarity attributes
	# Map user similarity attributes
	sideth2usth, usth2sideth = dict(), dict()
	num_ustypes = 0
	for sideth in range(len(entity_types)):
		first_entity_type = entity_types[sideth][0]
		second_entity_type = entity_types[sideth][1]
		if (first_entity_type == 'u') and (second_entity_type == 's'):
			sideth2usth[sideth] = num_ustypes
			usth2sideth[num_ustypes] = sideth
			num_ustypes += 1

	# Map item similarity attributes
	sideth2isth, isth2sideth = dict(), dict()
	num_istypes = 0
	for sideth in range(len(entity_types)):
		first_entity_type = entity_types[sideth][0]
		second_entity_type = entity_types[sideth][1]
		if (first_entity_type == 'i') and (second_entity_type == 's'):
			sideth2isth[sideth] = num_istypes
			isth2sideth[num_istypes] = sideth
			num_istypes += 1

	return sideth2usth, usth2sideth, sideth2isth, isth2sideth, num_ustypes, num_istypes



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Metrics
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Measure precision@k and recall@k
def pre_recall_at_k(rui_s, hat_rui_s, k):
	Top_k_u = sorted(range(len(hat_rui_s)), key=lambda i: hat_rui_s[i])[-k:]
	Test_u = sorted(range(len(rui_s)), key=lambda i: rui_s[i])[-k:]
	
	if len(Test_u) == 0:
		return 0, 0
	intersects = [item for item in Top_k_u if item in Test_u]
	precision = len(intersects) / k
	recall = len(intersects) / len(Test_u)
	return precision, recall


# Measure Spearman's rho
def corr(rui_s, hat_rui_s):
	if len(rui_s) == 1:
		return 1

	# Get o_ui^*, h_ui^* dict
	rui_s_cp = np.copy(rui_s)
	o_dict, o_num = avg_rank_dict(rui_s_cp)
	if len(o_dict) == 1:
		return 1
	hat_rui_s_cp = np.copy(hat_rui_s)
	h_dict, h_num = avg_rank_dict(hat_rui_s_cp)

	# Get o_bar and h_bar
	o_bar = bar_rank(o_dict, o_num)
	h_bar = bar_rank(h_dict, h_num)

	# Get rho_u
	numer1 = std_dev_rank(o_dict, o_num, o_bar)
	numer2 = std_dev_rank(h_dict, h_num, h_bar)
	denom = 0
	for i in range(len(rui_s)):
		o_i = rui_s[i]
		h_i = hat_rui_s[i]
		rank_o = o_dict[o_i]
		rank_h = h_dict[h_i]
		denom += (rank_o - o_bar) * (rank_h - h_bar)
	rho_u = denom / (numer1 * numer2)

	return rho_u


# Generate a dictionary for the average rank of the same ranked items
def avg_rank_dict(lst):
	# input: lst
	# output: a dict
	# - key: a rating
	# - value: average rank of items whose rating is key
	avg_dict = dict()
	num_dict = dict()
	lst.sort()
	target_rating = lst[0]
	sum_rank = 0
	num = 0
	for th, rating in enumerate(lst):
		if rating > target_rating:
			avg_dict[target_rating] = sum_rank / num
			num_dict[target_rating] = num
			target_rating = rating
			sum_rank = th + 1
			num = 1
		else:
			sum_rank += th + 1
			num += 1
	avg_dict[target_rating] = sum_rank / num
	num_dict[target_rating] = num

	return avg_dict, num_dict


# Calculate the average rank of the same ranked items
def bar_rank(rank_dict, num_dict):
	total_rank = 0
	total_num = 0
	for rating in rank_dict:
		rank = rank_dict[rating]
		num = num_dict[rating]
		total_rank += rank * num
		total_num += num
	return total_rank / total_num


# Calculate standard deviation of rankings
def std_dev_rank(rank_dict, num_dict, bar):
	sum_dev_sq = 0
	for e in rank_dict:
		avg_rank = rank_dict[e]
		num_same_rating = num_dict[e]
		sum_dev_sq += ((avg_rank - bar) ** 2) * num_same_rating
	return np.sqrt(sum_dev_sq)



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# etc.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Return linearly scaled value of x
#  in [a, b] -> [m, M] scaling
def lin_scale(x, a, b, m, M):
	p = (M - m) / (b - a)
	q = M - p * b
	return p * x + q

	