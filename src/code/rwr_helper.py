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
- rwr_helper.py
  : includes helper functions for random walk with restart 
    methods.


This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import packages
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
from entire_helper import *


""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Sample seed users
""""""""""""""""""""""""""""""""""""""""""""""""""""""
def sample_seed_users(train, test, num_sample):
	# Find out common users who are in both training and test set
	common_users = dict()
	for u in train.u:
		if u in test.u:
			u = int(u)
			common_users[u] = True
	
	# Randomly sample seed users from the common_users
	seedu2th = dict()
	th2seedu = dict()
	num_common_users = len(common_users)
	sample_ratio = num_sample / num_common_users
	num_seed_user = 0
	for u in common_users:
		rand_num = np.random.random()
		if rand_num < sample_ratio:
			seedu2th[u] = num_seed_user
			th2seedu[num_seed_user] = u
			num_seed_user += 1

	return seedu2th, th2seedu


""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Make \tilde{A}, a normalized adj matrix
""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Generate \tilde{A} with explicit data
def get_adj_dict(e1s, e2s, vs, A):
	# Get adj table
	for e1, e2, v in zip(e1s, e2s, vs):
		e1 = int(e1)
		e2 = int(e2)
		if not (e1 in A):
			A[e1] = list()
		if not (e2 in A):
			A[e2] = list()
		A[e1].append([e2, v])
		A[e2].append([e1, v])
	return A

# Generate \tilde{A} with implicit data
def get_imp_adj_dict(e1s, e2s, vs, alpha, A):
	# Get adj table
	for e1, e2, v in zip(e1s, e2s, vs):
		e1 = int(e1)
		e2 = int(e2)
		if not (e1 in A):
			A[e1] = list()
		if not (e2 in A):
			A[e2] = list()
		v = 1 + alpha * v
		A[e1].append([e2, v])
		A[e2].append([e1, v])
	return A

# Generate a dictionary whose key is entity
#  and value is sum of nnz value (e.g. rating) of the entity
def get_sum_dict(A):
	sum_dict = dict()
	for e in A:
		A[e] = np.array(A[e])
		sum_e = np.sum(A[e][:, 1])
		sum_dict[e] = sum_e
	return sum_dict
	

# Normalize adjacent matrix A
def normalize_adj(A, sum_dict):
	for e in A:
		for th, (e_nei, r) in enumerate(A[e]):
			A[e][th] = [e_nei, r / sum_dict[e_nei]]


""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Make b, a bias term
""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Generate a vector m for bias term
def gen_m(A, sum_dict):
	m_dict = dict()
	sum_mus = 0
	for e in A:
		mu_e = sum_dict[e] / len(A[e])
		m_dict[e] = mu_e
		sum_mus += mu_e
	for e in m_dict:
		m_dict[e] /= sum_mus
	return m_dict


""""""""""""""""""""""""""""""""""""""""""""""""""""""
# RWR
""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Get inner product of A[e] and vec
def get_inner_product(eth, A, e2th, th2e, vec):
	e = th2e[eth]
	nei_scores = A[e]
	inn = 0
	for nei, score in nei_scores:
		inn += score * vec[e2th[nei]]
	return inn


""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Measure Spearman's rho, precision@k, and recall@k
""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Measure the metrics without sampling of seed users
def rwr_measure(args, fold, paras):
	# 0. Basic settings
	k = args.k
	mtd = args.method
	dataset = args.dataset
	result_path = args.result_path

	# 1. Read ratings, rank matrix, and entity mappings
	RWR_filename = result_path + '%s_%s_score_%s.txt' % (mtd, dataset, paras)
	item_filename = result_path + '%s_%s_item_%s.txt' % (mtd, dataset, paras)
	seed_user_filename = result_path + '%s_%s_seed_user_%s.txt' % (mtd, dataset, paras)
	train, test = read_rating(args.input_path, fold)
	rank_matrix = np.loadtxt(RWR_filename)
	item_maps = pd.read_csv(item_filename, sep='\t', names=['i'])
	user_maps = pd.read_csv(seed_user_filename, sep='\t', names=['u'])

	# 2. Make test rating table
	test_table = get_test_table(test)

	# 3. Map entities
	i2th, th2i, dummy, num_i = map_entity(item_maps.i, dict(), dict(), 0, 0)
	u2th, th2u, dummy, num_u = map_entity(user_maps.u, dict(), dict(), 0, 0)
	
	# 4. Check seed users
	num_seedu = 0
	for seed_u in u2th:
		if seed_u in test_table:
			num_seedu += 1

	# 5. Measure correlations, precision@k, and recall@k
	corrs = np.zeros(num_seedu)
	patks = np.zeros(num_seedu)
	ratks = np.zeros(num_seedu)

	seed_u_th = 0
	for seed_u in u2th:
		if not(seed_u in test_table):
			continue

		# Get RWR scores w.r.t seed_u
		num_items_of_seed_u = len(test_table[seed_u])
		obs = np.zeros(num_items_of_seed_u)
		prs = np.zeros(num_items_of_seed_u)
		for ith, (i, r) in enumerate(test_table[seed_u]):
			# If the item i is not learned in the training set
			if not (i in i2th):
				# i's rwr score should be 0
				continue

			rwr_score = rank_matrix[u2th[seed_u]][i2th[i]]
			obs[ith] = r
			prs[ith] = rwr_score

		# Get spearmans rho of the seed user
		rho = corr(obs, prs)
		corrs[seed_u_th] = rho

		# Get precision and recall @ k of the seed user
		precision, recall = pre_recall_at_k(obs, prs, k)
		patks[seed_u_th] = precision
		ratks[seed_u_th] = recall
		seed_u_th += 1

	return np.average(corrs), np.average(patks), np.average(ratks)


# Measure the metrics with sampling of seed users
def rwr_measure_seed_sample(args, fold, paras):
	# 0. Basic settings
	k = args.k
	mtd = args.method
	dataset = args.dataset
	result_path = args.result_path

	# 1. Read ratings, rank matrix, and entity mappings
	RWR_filename = result_path + '%s_%s_score_%s.txt' % (mtd, dataset, paras)
	item_filename = result_path + '%s_%s_item_%s.txt' % (mtd, dataset, paras)
	seed_user_filename = result_path + '%s_%s_seed_user_%s.txt' % (mtd, dataset, paras)
	train, test = read_rating(args.input_path, fold)
	rank_matrix = np.loadtxt(RWR_filename)
	item_maps = pd.read_csv(item_filename, sep='\t', names=['i'])
	user_maps = pd.read_csv(seed_user_filename, sep='\t', names=['u'])

	# 2. Make test rating table
	test_table = get_test_table(test)

	# 3. Map entities
	i2th, th2i, dummy, num_i = map_entity(item_maps.i, dict(), dict(), 0, 0)
	seedu2th, th2seedu, dummy, num_seedu = map_entity(user_maps.u, dict(), dict(), 0, 0)

	# 4. Check seed users
	num_seedu = 0
	for seed_u in seedu2th:
		if seed_u in test_table:
			num_seedu += 1

	# 5. Measure correlations, precision@k, and recall@k
	corrs = np.zeros(num_seedu)
	patks = np.zeros(num_seedu)
	ratks = np.zeros(num_seedu)
	seed_u_th = 0
	for seed_u in seedu2th:
		if not(seed_u in test_table):
			continue

		# Get RWR scores w.r.t seed_u
		num_items_of_seed_u = len(test_table[seed_u])
		obs = np.zeros(num_items_of_seed_u)
		prs = np.zeros(num_items_of_seed_u)
		for ith, (i, r) in enumerate(test_table[seed_u]):
			# If the item i is not learned in the training set
			if not (i in i2th):
				# i's rwr score should be 0
				continue

			rwr_score = rank_matrix[seedu2th[seed_u]][i2th[i]]
			obs[ith] = r
			prs[ith] = rwr_score

		# Get spearmans rho of the seed user
		rho = corr(obs, prs)
		corrs[seed_u_th] = rho

		# Get precision and recall @ k of the seed user
		precision, recall = pre_recall_at_k(obs, prs, k)
		patks[seed_u_th] = precision
		ratks[seed_u_th] = recall
		seed_u_th += 1

	return np.average(corrs), np.average(patks), np.average(ratks)


