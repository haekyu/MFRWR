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
- mf_helper.py
  : includes helper functions for matrix factorization.


This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import packages 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
from entire_helper import *


""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Measure rmse and mae
""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Measure rmse and mae for given explicit ratings
def mf_exp_accuracy(dat, X, Y, u2th, i2th, mu):
	# Measure RMSE and MAE
	prediction = list()
	for u, i in zip(dat.u, dat.i):
		u = int(u)
		i = int(i)
		if (u in u2th) and (i in i2th):
			user = u2th[u]
			item = i2th[i]
			hat_r_ui = np.matmul(X[user], Y[item])
		else:
			hat_r_ui = mu
		prediction.append(hat_r_ui)
	RMSE = np.sqrt(np.average(np.subtract(prediction, dat.r) ** 2))
	MAE = np.average(np.absolute(np.subtract(prediction, dat.r)))

	return RMSE, MAE

# Measure rmse and mae for given implicit ratings
# (only used in checking convergence of learning)
def mf_imp_accuracy(dat, X, Y, u2th, i2th, mu):
	# Ground rating
	ratings = np.ones(len(train))

	# Measure RMSE and MAE
	prediction = list()
	for u, i in zip(dat.u, dat.i):
		u = int(u)
		i = int(i)
		if (u in u2th) and (i in i2th):
			user = u2th[u]
			item = i2th[i]
			hat_r_ui = np.matmul(X[user], Y[item])
		else:
			hat_r_ui = mu
		prediction.append(hat_r_ui)
	RMSE = np.sqrt(np.average(np.subtract(prediction, ratings) ** 2))
	MAE = np.average(np.absolute(np.subtract(prediction, ratings)))

	return RMSE, MAE

# Measure rmse and mae for MF_bias
def mf_bias_accuracy(dat, X, Y, B_u, B_i, u2th, i2th, mu):
	# Measure RMSE and MAE
	prediction = list()
	for u, i in zip(dat.u, dat.i):
		u = int(u)
		i = int(i)
		if (u in u2th) and (i in i2th):
			user = u2th[u]
			item = i2th[i]
			hat_r_ui = mu + B_u[user] + B_i[item] + np.matmul(X[user], Y[item])
		else:
			hat_r_ui = mu
		prediction.append(hat_r_ui)
	RMSE = np.sqrt(np.average(np.subtract(prediction, dat.r) ** 2))
	MAE = np.average(np.absolute(np.subtract(prediction, dat.r)))

	return RMSE, MAE


""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Measure Spearman's rho, precision@k, and recall@k
""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Measure the metrics when bias terms are not included
def mf_unbiased_measure(args, fold, paras):
	# 0. Basic settings
	k = args.k
	mtd = args.method
	dataset = args.dataset
	result_path = args.result_path
	
	# 1. Read ratings, vectors, and entity mappings
	train, test = read_rating(args.input_path, fold)
	X = np.loadtxt(result_path + '%s_%s_X_%s.txt' % (mtd, dataset, paras), delimiter='\t')
	Y = np.loadtxt(result_path + '%s_%s_Y_%s.txt' % (mtd, dataset, paras), delimiter='\t')
	users = np.loadtxt(result_path + '%s_%s_user_%s.txt' % (mtd, dataset, paras))
	items = np.loadtxt(result_path + '%s_%s_item_%s.txt' % (mtd, dataset, paras))

	# 2. Map entities
	u2th, th2u, _, num_u = map_entity(users, dict(), dict(), 0, 0)
	i2th, th2i, _, num_i = map_entity(items, dict(), dict(), 0, 0)

	# 3. Get rating tables
	test_table = get_test_table(test)

	# 4. Check the number of users to test
	num_test_user = 0
	for train_u in range(num_u):
		train_u = th2u[train_u]
		train_u = int(train_u)
		if train_u in test_table:
			num_test_user += 1

	# 5. Measure correlations, precision@k, and recall@k
	corrs = np.zeros(num_test_user)
	patks = np.zeros(num_test_user)
	ratks = np.zeros(num_test_user)
	test_user_th = 0
	for u in range(num_u):
		# Get u
		u = th2u[u]
		u = int(u)

		if not (u in test_table):
			continue

		# Get u's observed and predicted ratings
		num_ratings = len(test_table[u])
		obs = np.zeros(num_ratings)
		prs = np.zeros(num_ratings)
		for j, (i_of_u, r_of_u) in enumerate(test_table[u]):
			# If the item i_of_u is not learned in the training set
			if not (i_of_u in i2th):
				# i_of_u's score should be 0
				continue
			hat_r_ui = np.matmul(X[u2th[u]], Y[i2th[i_of_u]])
			obs[j] = r_of_u
			prs[j] = hat_r_ui

		# Get spearmans rho of u
		rho = corr(obs, prs)
		corrs[test_user_th] = rho

		# Get precision and recall @ k of u
		precision, recall = pre_recall_at_k(obs, prs, k)
		patks[test_user_th] = precision
		ratks[test_user_th] = recall
		test_user_th += 1

	return np.average(corrs), np.average(patks), np.average(ratks)

# Measure the metrics when bias terms are included
def mf_bias_measure(args, fold, paras):
	# 0. Basic settings
	k = args.k
	mtd = args.method
	dataset = args.dataset
	result_path = args.result_path
	
	# 1. Read ratings, vectors, and entity mappings
	train, test = read_rating(args.input_path, fold)
	X = np.loadtxt(result_path + '%s_%s_X_%s.txt' % (mtd, dataset, paras), delimiter='\t')
	Y = np.loadtxt(result_path + '%s_%s_Y_%s.txt' % (mtd, dataset, paras), delimiter='\t')
	B_u = np.loadtxt(result_path + '%s_%s_Bu_%s.txt' % (mtd, dataset, paras), delimiter='\t')
	B_i = np.loadtxt(result_path + '%s_%s_Bi_%s.txt' % (mtd, dataset, paras), delimiter='\t')
	users = np.loadtxt(result_path + '%s_%s_user_%s.txt' % (mtd, dataset, paras))
	items = np.loadtxt(result_path + '%s_%s_item_%s.txt' % (mtd, dataset, paras))
	mu = np.average(train.r)

	# 2. Map entities
	u2th, th2u, _, num_u = map_entity(users, dict(), dict(), 0, 0)
	i2th, th2i, _, num_i = map_entity(items, dict(), dict(), 0, 0)

	# 3. Get rating tables
	test_table = get_test_table(test)

	# 4. Check the number of users to test
	num_test_user = 0
	for train_u in range(num_u):
		train_u = th2u[train_u]
		train_u = int(train_u)
		if train_u in test_table:
			num_test_user += 1

	# 5. Measure correlations, precision@k, and recall@k
	corrs = np.zeros(num_test_user)
	patks = np.zeros(num_test_user)
	ratks = np.zeros(num_test_user)
	test_user_th = 0
	for u in range(num_u):
		# Get u
		u = th2u[u]
		u = int(u)

		if not (u in test_table):
			continue

		# Get u's observed and predicted ratings
		num_ratings = len(test_table[u])
		obs = np.zeros(num_ratings)
		prs = np.zeros(num_ratings)
		for j, (i_of_u, r_of_u) in enumerate(test_table[u]):
			# If the item i_of_u is not learned in the training set
			if not (i_of_u in i2th):
				# i_of_u's score should be 0
				continue
			user = u2th[u]
			item = i2th[i_of_u]
			hat_r_ui = mu + B_u[user] + B_i[item] + np.matmul(X[user], Y[item])
			obs[j] = r_of_u
			prs[j] = hat_r_ui

		# Get spearmans rho of u
		rho = corr(obs, prs)
		corrs[test_user_th] = rho

		# Get precision and recall @ k of u
		precision, recall = pre_recall_at_k(obs, prs, k)
		patks[test_user_th] = precision
		ratks[test_user_th] = recall
		test_user_th += 1

	return np.average(corrs), np.average(patks), np.average(ratks)
