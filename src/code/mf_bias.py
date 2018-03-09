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
- mf_bias.py
  : matrix factorization with bias terms.

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import packages 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from entire_helper import *
from mf_helper import *


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# MF_bias 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def mf_bias(args, fold):
	# 0. Basic settings
	lr = args.lr
	lamb = args.lamb
	mtd = args.method
	dim = args.dim
	dataset = args.dataset
	result_path = args.result_path

	# 1. Read ratings
	train, test = read_rating(args.input_path, fold)
	mu = np.average(train.r)
	maxR = max(train.r)
	minR = min(train.r)

	# 2. Mapping entities
	u2th, th2u, _, num_u = map_entity(train.u, dict(), dict(), 0, 0)
	i2th, th2i, _, num_i = map_entity(train.i, dict(), dict(), 0, 0)

	# 3. Initialize vectors and biases of users and items
	a = np.sqrt((maxR + minR) / dim)
	X = (np.random.random((num_u, dim))) * a
	Y = (np.random.random((num_i, dim))) * a
	B_u = (np.random.random(num_u)) * a
	B_i = (np.random.random(num_i)) * a

	# 4. Learn
	prev_rmse = 10
	num_iter = 0
	max_iter = 30
	while num_iter < max_iter:
		num_iter += 1

		# Update latent features
		for u, i, r in zip(train.u, train.i, train.r):
			user = u2th[int(u)]
			item = i2th[int(i)]
			hat_r_ui = mu + B_u[user] + B_i[item] + np.matmul(X[user], Y[item])
			err = r - hat_r_ui
			X[user] += lr * (err * Y[item] - lamb * X[user])
			Y[item] += lr * (err * X[user] - lamb * Y[item])

		# Check train rmse with train examples
		rmse, mae = mf_bias_accuracy(train, X, Y, B_u, B_i, u2th, i2th, mu)

		# Check convergence
		if prev_rmse < rmse:
			break
		if np.absolute(np.subtract(prev_rmse, rmse)) < 0.001:
			break
		prev_rmse = rmse

	# 5. Test RMSE and MAE
	testRMSE, testMAE = mf_bias_accuracy(test, X, Y, B_u, B_i, u2th, i2th, mu)

	# 6. Save results
	paras = '%s_%s_%s_%s' % (dim, lr, lamb, fold)

	# Save vectors
	np.savetxt(result_path + '%s_%s_X_%s.txt' % (mtd, dataset, paras), X, fmt='%.5f', delimiter='\t')
	np.savetxt(result_path + '%s_%s_Y_%s.txt' % (mtd, dataset, paras), Y, fmt='%.5f', delimiter='\t')

	# Save biases
	np.savetxt(result_path + '%s_%s_Bu_%s.txt' % (mtd, dataset, paras), B_u, fmt='%.5f', delimiter='\t')
	np.savetxt(result_path + '%s_%s_Bi_%s.txt' % (mtd, dataset, paras), B_i, fmt='%.5f', delimiter='\t')

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



