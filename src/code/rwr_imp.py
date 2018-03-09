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
- rwr_imp.py
  : random walk with restart for implicit ratings


This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import packages
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from rwr_helper import *
from entire_helper import *


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# RWR_imp
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# RWR_imp without sampling seed users
def rwr_imp(args, fold):
	# 0. Basic settings
	c = args.c
	mtd = args.method
	alpha = args.alpha
	dataset = args.dataset
	result_path = args.result_path

	# 1. Read ratings
	train, test = read_rating(args.input_path, fold)

	# 2. Mapping entities
	e2th, th2e, num_e, num_u = map_entity(train.u, dict(), dict(), 0, 0)
	e2th, th2e, num_e, num_i = map_entity(train.i, e2th, th2e, 0, num_e)

	# 3. Generate adjacency matrix A which is column normalized
	A = get_imp_adj_dict(train.u, train.i, train.r, alpha, dict())
	sum_dict = get_sum_dict(A)
	normalize_adj(A, sum_dict)

	# 4. Get RWR scores w.r.t all users
	r_matrix = np.zeros((num_u, num_i))
	i_st, i_end = num_u, num_e

	for seed_u_th in range(num_u):
		# Set vectors initially
		q = np.zeros(num_e)
		q[seed_u_th] = 1
		prev_r = np.array([1 / num_e] * num_e)
		r = np.array([1 / num_e] * num_e)

		# RWR (Restart from the seed user)
		while True:
			# Update rank vector
			for eth in range(num_e):
				inn = get_inner_product(eth, A, e2th, th2e, prev_r)
				r[eth] = (1 - c) * inn + c * q[eth]

			# Check convergence
			diff = np.linalg.norm(prev_r - r)
			if diff < 0.0001:
				break

			# Update prev rank vector
			prev_r = np.copy(r)

		# Save RWR scores w.r.t the seed user
		r_matrix[seed_u_th] = np.copy(r[i_st: i_end])

	# 5. Save the results
	paras = '%s_%s_%s' % (c, alpha, fold)

	# Save mapping of seed users
	f = open(result_path + '%s_%s_seed_user_%s.txt' % (mtd, dataset, paras), 'w')
	for u_th in range(num_u):
		seed_u = th2e[u_th]
		f.write('%d\n' % seed_u)
	f.close()

	# Save mapping of items
	f = open(result_path + '%s_%s_item_%s.txt' % (mtd, dataset, paras), 'w')
	for i_th in range(num_i):
		i_id = th2e[num_u + i_th]
		f.write('%d\n' % i_id)
	f.close()

	# Save rank matrix
	RWR_filename = result_path + '%s_%s_score_%s.txt' % (mtd, dataset, paras)
	np.savetxt(RWR_filename, r_matrix)


# RWR_imp with sampling seed users
def rwr_imp_seed_sample(args, fold):
	# 0. Basic settings
	c = args.c
	mtd = args.method
	alpha = args.alpha
	dataset = args.dataset
	result_path = args.result_path
	
	# 1. Read ratings
	train, test = read_rating(args.input_path, fold)

	# 2. Mapping entities
	e2th, th2e, num_e, num_u = map_entity(train.u, dict(), dict(), 0, 0)
	e2th, th2e, num_e, num_i = map_entity(train.i, e2th, th2e, 0, num_e)

	# 3. Sample seed users - XXX with sampling seed users XXX
	seedu2th, th2seedu = sample_seed_users(train, test, args.num_seed)
	num_seed_u = len(seedu2th)

	# 4. Generate adjacency matrix A which is column normalized
	A = get_imp_adj_dict(train.u, train.i, train.r, alpha, dict())
	sum_dict = get_sum_dict(A)
	normalize_adj(A, sum_dict)

	# 5. Get RWR scores w.r.t all seed users
	#    - XXX only for sampled seed users XXX
	r_matrix = np.zeros((num_seed_u, num_i))
	i_st, i_end = num_u, num_e

	for seed_u in seedu2th: 
		# Set vectors initially
		q = np.zeros(num_e)
		q[e2th[seed_u]] = 1
		prev_r = np.array([1 / num_e] * num_e)
		r = np.array([1 / num_e] * num_e)

		# RWR (Restart from the seed user)
		while True:
			# Update rank vector
			for eth in range(num_e):
				inn = get_inner_product(eth, A, e2th, th2e, prev_r)
				r[eth] = (1 - c) * inn + c * q[eth]

			# Check convergence
			diff = np.linalg.norm(prev_r - r)
			if diff < 0.0001:
				break

			# Update prev rank vector
			prev_r = np.copy(r)

		# Save RWR scores w.r.t the seed user
		r_matrix[seedu2th[seed_u]] = np.copy(r[i_st: i_end])

	# 8. Save the results
	paras = 'sample_%s_%s_%s' % (c, alpha, fold)

	# Save mapping of seed users
	f = open(result_path + '%s_%s_seed_user_%s.txt' % (mtd, dataset, paras), 'w')
	for u_th in range(num_seed_u):
		seed_u = th2seedu[u_th]
		f.write('%d\n' % seed_u)
	f.close()

	# Save mapping of items
	f = open(result_path + '%s_%s_item_%s.txt' % (mtd, dataset, paras), 'w')
	for i_th in range(num_i):
		i_id = th2e[num_u + i_th]
		f.write('%d\n' % i_id)
	f.close()

	# Save rank matrix
	RWR_filename = result_path + '%s_%s_score_%s.txt' % (mtd, dataset, paras)
	np.savetxt(RWR_filename, r_matrix)

