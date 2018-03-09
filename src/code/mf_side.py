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
- mf_side.py
  : matrix factorization when side information is given


This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import packages 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from entire_helper import *
from mf_helper import *


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# MF_side
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def mf_side(args, fold):
	# 0. Basic settings
	lr = args.lr
	lamb = args.lamb
	mtd = args.method
	dim = args.dim
	dataset = args.dataset
	alpha = args.alpha
	result_path = args.result_path
	sides = args.side_paths
	entity_types = args.entity_types
	is_implicit = args.is_implicit

	# 1. Read ratings and side info
	train, test = read_rating(args.input_path, fold)
	side_bucket = list()
	for side in sides:
		side_info = pd.read_csv(side, sep='\t', names=['e1', 'e2'])
		side_bucket.append(side_info)

	# 2. Mapping entities and attributes
	u2th, th2u, i2th, th2i, num_u, num_i = map_ui_with_side(train, side_bucket, entity_types)
	sideth2usth, usth2sideth, sideth2isth, isth2sideth, num_us, num_is = map_attributes(side_bucket, entity_types)

	# 3. Initialize vectors of users and items
	maxR = max(train.r)
	minR = min(train.r)
	if is_implicit:
		paras = '%s_%s_%s_%s_%s' % (dim, lr, lamb, alpha, fold)
		X = (np.random.random((num_u, dim)))
		Y = (np.random.random((num_i, dim)))
		W = (np.random.random((num_us, dim)))
		Z = (np.random.random((num_is, dim)))
	else:
		paras = '%s_%s_%s_%s' % (dim, lr, lamb, fold)
		a = np.sqrt((maxR + minR) / dim)
		X = (np.random.random((num_u, dim))) * a
		Y = (np.random.random((num_i, dim))) * a
		W = (np.random.random((num_us, dim))) * a
		Z = (np.random.random((num_is, dim))) * a

	# 4. Learn
	prev_rmse = 10
	num_iter = 0
	max_iter = 30
	while num_iter < max_iter:
		num_iter += 1

		# Factorization of ratings
		for u, i, r in zip(train.u, train.i, train.r):
			user = u2th[int(u)]
			item = i2th[int(i)]
			hat_r_ui = np.matmul(X[user], Y[item])
			if is_implicit:
				c_ui = 1 + alpha * r
				err = 1 - hat_r_ui
				X[user] += lr * (c_ui * err * Y[item] - lamb * X[user])
				Y[item] += lr * (c_ui * err * X[user] - lamb * Y[item])
			else:
				err = r - hat_r_ui
				X[user] += lr * (err * Y[item] - lamb * X[user])
				Y[item] += lr * (err * X[user] - lamb * Y[item])

		# Factorization of user similarity attributes
		for sideth in range(len(entity_types)):
			first_entity_type = entity_types[sideth][0]
			second_entity_type = entity_types[sideth][1]
			if (first_entity_type == 'u') and (second_entity_type == 's'):
				for u, a in zip(side[sideth].e1, side[sideth].e2):
					user = u2id[int(u)]
					us = sideth2usth[sideth]
					minA = min(side[sideth].e2)
					maxA = max(side[sideth].e2)
					if is_implicit:
						a = lin_scale(a, minA, maxA, 1 + alpha * minR, 1 + alpha * maxR)
					else:
						a = lin_scale(a, minA, maxA, minR, maxR)
					hat_a_us = np.matmul(X[user], W[us])
					err = a - hat_a_us
					X[user] += lr * (err * W[us] - lamb * X[user])
					W[us] += lr * (err * X[user] - lamb * W[us])

		# Factorization of item similarity attributes
		for sideth in range(len(entity_types)):
			first_entity_type = entity_types[sideth][0]
			second_entity_type = entity_types[sideth][1]
			if (first_entity_type == 'i') and (second_entity_type == 's'):
				for i, b in zip(side[sideth].e1, side[sideth].e2):
					item = i2id[int(i)]
					items = sideth2isth[sideth]
					minA = min(side[sideth].e2)
					maxA = max(side[sideth].e2)
					if is_implicit:
						b = lin_scale(b, minA, maxA, 1 + alpha * minR, 1 + alpha * maxR)
					else:
						b = lin_scale(b, minA, maxA, minR, maxR)
					hat_b_us = np.matmul(Y[item], Z[items])
					err = b - hat_b_us
					Y[item] += lr * (err * Z[items] - lamb * Y[item])
					Z[items] += lr * (err * Y[item] - lamb * Z[items])

		# Check train rmse with train examples
		rmse, mae = mf_exp_accuracy(train, X, Y, u2th, i2th, mu)

		# Check convergence
		if prev_rmse < rmse:
			break
		if np.absolute(np.subtract(prev_rmse, rmse)) < 0.001:
			break
		prev_rmse = rmse

	# 5. Test RMSE and MAE
	testRMSE, testMAE = mf_unbiased_accuracy(test, X, Y, u2th, i2th, mu)

	# 6. Save results

	# Save vectors
	np.savetxt(result_path + '%s_%s_X_%s.txt' % (mtd, dataset, paras), X, fmt='%.5f', delimiter='\t')
	np.savetxt(result_path + '%s_%s_Y_%s.txt' % (mtd, dataset, paras), Y, fmt='%.5f', delimiter='\t')

	# Save mapping of users
	f = open(result_path + '%s_%s_user_%s.txt' % (mtd, dataset, paras), 'w')
	for u_th in range(num_u):
		u = th2u[u_th]
		f.write('%d\n' % u)
	f.close()

	# Save mapping of items
	f = open(result_path + '%s_%s_item_%s.txt' % (mtd, dataset, paras), 'w')
	for i_th in range(num_i):
		i = th2i[i_th]
		f.write('%d\n' % i)
	f.close()

	return rmse, testRMSE, testMAE


def mf_side_social(args, fold):
	# 0. Basic settings
	lr = args.lr
	lamb = args.lamb
	mtd = args.method
	dim = args.dim
	dataset = args.dataset
	alpha = args.alpha
	result_path = args.result_path
	sides = args.side_paths
	entity_types = args.entity_types
	is_implicit = args.is_implicit

	# 1. Read ratings and side info
	train, test = read_rating(args.input_path, fold)
	side_bucket = list()
	for side in sides:
		side_info = pd.read_csv(side, sep='\t', names=['e1', 'e2'])
		side_bucket.append(side_info)

	# 2. Mapping entities and attributes
	u2th, th2u, i2th, th2i, num_u, num_i = map_ui_with_side(train, side_bucket, entity_types)

	# 3. Initialize vectors of users and items
	if is_implicit:
		paras = '%s_%s_%s_%s_%s' % (dim, lr, lamb, alpha, fold)
		X = (np.random.random((num_u, dim)))
		Y = (np.random.random((num_i, dim)))
		W = (np.random.random((num_u, dim)))
	else:
		paras = '%s_%s_%s_%s' % (dim, lr, lamb, fold)
		maxR = max(train.r)
		minR = min(train.r)	
		a = np.sqrt((maxR + minR) / dim)
		X = (np.random.random((num_u, dim))) * a
		Y = (np.random.random((num_i, dim))) * a
		W = (np.random.random((num_u, dim))) * a

	# 4. Learn
	prev_rmse = 10
	num_iter = 0
	max_iter = 30
	while num_iter < max_iter:
		num_iter += 1

		# Factorization of ratings
		for u, i, r in zip(train.u, train.i, train.r):
			user = u2th[int(u)]
			item = i2th[int(i)]
			hat_r_ui = np.matmul(X[user], Y[item])
			if is_implicit:
				c_ui = 1 + alpha * r
				err = 1 - hat_r_ui
				X[user] += lr * (c_ui * err * Y[item] - lamb * X[user])
				Y[item] += lr * (c_ui * err * X[user] - lamb * Y[item])
			else:
				err = r - hat_r_ui
				X[user] += lr * (err * Y[item] - lamb * X[user])
				Y[item] += lr * (err * X[user] - lamb * Y[item])

		# Factorization of social links
		for sideth in range(len(entity_types)):
			first_entity_type = entity_types[sideth][0]
			second_entity_type = entity_types[sideth][1]
			if (first_entity_type == 'u') and (second_entity_type == 'u'):
				for u1, u2 in zip(side[sideth].e1, side[sideth].e2):
					u1 = u2id[int(u1)]
					u2 = u2id[int(u2)]
					err = 1 - np.matmul(X[u1], W[u2])
					X[u1] += lr * (err * W[u2] - lamb * X[u1])
					W[u2] += lr * (err * X[u1] - lamb * W[u2])
					err = 1 - np.matmul(X[u2], W[u1])
					X[u2] += lr * (err * W[u1] - lamb * X[u2])
					W[u1] += lr * (err * X[u2] - lamb * W[u1])

		# Check train rmse with train examples
		rmse, mae = mf_exp_accuracy(train, X, Y, u2th, i2th, mu)

		# Check convergence
		if prev_rmse < rmse:
			break
		if np.absolute(np.subtract(prev_rmse, rmse)) < 0.001:
			break
		prev_rmse = rmse

	# 5. Test RMSE and MAE
	testRMSE, testMAE = mf_unbiased_accuracy(test, X, Y, u2th, i2th, mu)

	# 6. Save results

	# Save vectors
	np.savetxt(result_path + '%s_%s_X_%s.txt' % (mtd, dataset, paras), X, fmt='%.5f', delimiter='\t')
	np.savetxt(result_path + '%s_%s_Y_%s.txt' % (mtd, dataset, paras), Y, fmt='%.5f', delimiter='\t')

	# Save mapping of users
	f = open(result_path + '%s_%s_user_%s.txt' % (mtd, dataset, paras), 'w')
	for u_th in range(num_u):
		u = th2u[u_th]
		f.write('%d\n' % u)
	f.close()

	# Save mapping of items
	f = open(result_path + '%s_%s_item_%s.txt' % (mtd, dataset, paras), 'w')
	for i_th in range(num_i):
		i = th2i[i_th]
		f.write('%d\n' % i)
	f.close()

	return rmse, testRMSE, testMAE

