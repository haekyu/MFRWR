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
- main.py
  : runs each scenarios of recommendation.

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import packages
""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import sys
import argparse
from mf_exp import *
from rwr_exp import *
from rwr_imp import *
from rwr_bias import *
from rwr_side import *
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))


# Main function
def main():
	dataset = 'filmtrust'
	args = parse_args(dataset)

	if args.config:
		print('Read parameters from %s.' % args.config_path)
		config_reader(args)
	
	# Split 5 folds
	if args.method == 'split5folds':
		split_5_folds(args)

	# MF
	# MF_exp
	if args.method == 'MF_exp':
		print_args(args)
		for fold in range(5):
			trainRMSE, testRMSE, testMAE = mf_exp(args, fold)
			paras = '%s_%s_%s_%s' % (args.dim, args.lr, args.lamb, fold)
			corr, patk, ratk = mf_unbiased_measure(args, fold, paras)
			print(corr, patk, ratk)

	# MF_imp
	elif args.method == 'MF_imp':
		print_args(args)
		for fold in range(5):
			trainRMSE, testRMSE, testMAE = mf_imp(args, fold)
			paras = '%s_%s_%s_%s_%s' % (args.dim, args.lr, args.lamb, args.alpha, fold)
			corr, patk, ratk = mf_unbiased_measure(args, fold, paras)
			print(corr, patk, ratk)

	# MF_bias
	elif args.method == 'MF_bias':
		print_args(args)
		for fold in range(5):
			trainRMSE, testRMSE, testMAE = mf_bias(args, fold)
			paras = '%s_%s_%s_%s' % (args.dim, args.lr, args.lamb, fold)
			corr, patk, ratk = mf_bias_measure(args, fold, paras)
			print(corr, patk, ratk)

	# MF_side
	elif args.method == 'MF_side':
		print_args(args)
		if args.is_social:
			for fold in range(5):
				trainRMSE, testRMSE, testMAE = mf_side_social(args, fold)
				if args.is_implicit:
					paras = '%s_%s_%s_%s_%s' % (args.dim, args.lr, args.lamb, args.alpha, fold)
				else:
					paras = '%s_%s_%s_%s' % (args.dim, args.lr, args.lamb, fold)
				corr, patk, ratk = mf_unbiased_measure(args, fold, paras)
				print(corr, patk, ratk)
		else:
			for fold in range(5):
				trainRMSE, testRMSE, testMAE = mf_side(args, fold)
				if args.is_implicit:
					paras = '%s_%s_%s_%s_%s' % (args.dim, args.lr, args.lamb, args.alpha, fold)
				else:
					paras = '%s_%s_%s_%s' % (args.dim, args.lr, args.lamb, fold)
				corr, patk, ratk = mf_unbiased_measure(args, fold, paras)
				print(corr, patk, ratk)

	# RWR
	# RWR_exp
	if args.method == 'RWR_exp':
		if args.is_sample:
			print_args(args)
			for fold in range(5):
				# rwr_exp_seed_sample(args, fold)
				paras = 'sample_%s_%s' % (args.c, fold)
				corr, patk, ratk = rwr_measure_seed_sample(args, fold, paras)
				print(corr, patk, ratk)
		else:
			print_args(args)
			for fold in range(5):
				rwr_exp(args, fold)
				paras = '%s_%s' % (args.c, fold)
				corr, patk, ratk = rwr_measure(args, fold, paras)
				print(corr, patk, ratk)

	# RWR_imp
	elif args.method == 'RWR_imp':
		if args.is_sample:
			print_args(args)
			for fold in range(5):
				rwr_imp_seed_sample(args, fold)
				paras = 'sample_%s_%s_%s' % (args.c, args.alpha, fold)
				corr, patk, ratk = rwr_measure_seed_sample(args, fold, paras)
				print(corr, patk, ratk)
		else:
			print_args(args)
			for fold in range(5):
				rwr_imp(args, fold)
				paras = '%s_%s_%s' % (args.c, args.alpha, fold)
				corr, patk, ratk = rwr_measure(args, fold, paras)
				print(corr, patk, ratk)

	# RWR_bias
	elif args.method == 'RWR_bias':
		if args.is_sample:
			print_args(args)
			for fold in range(5):
				rwr_bias_seed_sample(args, fold)
				paras = 'sample_%s_%s_%s_%s' % (args.c, args.beta, args.gamma, fold)
				corr, patk, ratk = rwr_measure_seed_sample(args, fold, paras)
				print(corr, patk, ratk)
		else:
			print_args(args)
			for fold in range(5):
				rwr_bias(args, fold)
				paras = '%s_%s_%s_%s' % (args.c, args.beta, args.gamma, fold)
				corr, patk, ratk = rwr_measure(args, fold, paras)
				print(corr, patk, ratk)

	# RWR_side
	elif args.method == 'RWR_side':
		if args.is_sample:
			print_args(args)
			for fold in range(5):
				rwr_side_seed_sample(args, fold)
				paras = 'sample_%s_%s_%s' % (args.c, args.delta, fold)
				corr, patk, ratk = rwr_measure_seed_sample(args, fold, paras)
				print(corr, patk, ratk)
		else:
			print_args(args)
			for fold in range(5):
				rwr_side(args, fold)
				paras = '%s_%s_%s' % (args.c, args.delta, fold)
				corr, patk, ratk = rwr_measure(args, fold, paras)
				print(corr, patk, ratk)


# Parse main method arguments
def parse_args(dataset='filmtrust'):
	# Create an argument parser
	parser = argparse.ArgumentParser('mfrwr')

	# Arguments for basic settings
	parser.add_argument('--dataset', default=dataset, help='Dataset name')
	parser.add_argument('--method', default='MF_exp', help='(MF/RWR)_(exp/imp/bias/side/cold) or split5folds')
	data_path = '../data/'
	parser.add_argument('--data_path', default=data_path, help='Input path')
	input_path = data_path + dataset + '/input/'
	parser.add_argument('--input_path', default=input_path, help='Input path')
	result_path = '../results/'
	parser.add_argument('--result_path', default=result_path, help='Input path')
	parser.add_argument('--side_paths', default=[input_path + 'link.tsv'], help='Files of side info')
	parser.add_argument('--entity_types', default=[['u', 'u']], help='Types of entity of side links')
	parser.add_argument('--config', default=False, help='Whether to give config file')
	parser.add_argument('--config_path', default='./myconfig.conf', help='Path of config file')

	# Arguments for both MF and RWR
	parser.add_argument('--alpha', default=0.001, help='coefficient of confidence level of implicit feedback')
	parser.add_argument('--is_implicit', default=False, help='whether the rating data is implicit')
	parser.add_argument('--is_social', default=False, help='whether social links are used')

	# Arguments for MF
	parser.add_argument('--lr', default=0.05, help='learning rate')
	parser.add_argument('--lamb', default=0.3, help='regularization parameter')
	parser.add_argument('--dim', default=5, help='dimension of vectors')

	# Arguments for RWR
	parser.add_argument('--is_sample', default=True, help='whether to sample seed users')
	parser.add_argument('--num_seed', default=300, help='# sampled seed users')
	parser.add_argument('--c', default=0.2, help='prob. of restart')
	parser.add_argument('--beta', default=0.4, help='prob. of walk in RWR_bias')
	parser.add_argument('--gamma', default=0.3, help='prob. of restart in RWR_bias')
	parser.add_argument('--delta', default=1.0, help='parameter for additional links in RWR_side')

	# Arguments for metric
	parser.add_argument('--k', default=1, help='# of top items for top-k prediction')	
	

	return parser.parse_args()


# Print arguments for each method
def print_args(args):
	# MF
	if args.method == 'MF_exp':
		prt_str = '%s, %s, dim:%s, lr:%s, lamb:%s, k:%s' % (args.method, args.dataset, args.dim, args.lr, args.lamb, args.k)
		print(prt_str)
	elif args.method == 'MF_imp':
		prt_str = '%s, %s, dim:%s, lr:%s, lamb:%s, alpha:%s, k:%s' % (args.method, args.dataset, args.dim, args.lr, args.lamb, args.alpha, args.k)
		print(prt_str)
	elif args.method == 'MF_bias':
		prt_str = '%s, %s, dim:%s, lr:%s, lamb:%s, k:%s' % (args.method, args.dataset, args.dim, args.lr, args.lamb, args.k)
		print(prt_str)
	elif args.method == 'MF_side':
		if args.is_implicit:
			if args.is_social:
				prt_str = '%s, %s, implicit, social, dim:%s, lr:%s, lamb:%s, alpha:%s, k:%s' % (args.method, args.dataset, args.dim, args.lr, args.lamb, args.alpha, args.k)
				print(prt_str)
			else:
				prt_str = '%s, %s, implicit, dim:%s, lr:%s, lamb:%s, alpha:%s, k:%s' % (args.method, args.dataset, args.dim, args.lr, args.lamb, args.alpha, args.k)
				print(prt_str)
		else:
			if args.is_social:
				prt_str = '%s, %s, social, dim:%s, lr:%s, lamb:%s, k:%s' % (args.method, args.dataset, args.dim, args.lr, args.lamb, args.k)
				print(prt_str)
			else:
				prt_str = '%s, %s, dim:%s, lr:%s, lamb:%s, k:%s' % (args.method, args.dataset, args.dim, args.lr, args.lamb, args.k)
				print(prt_str)

	# RWR
	if args.method == 'RWR_exp':
		if args.is_sample:
			prt_str = '%s, %s, #seedu:%d, c:%s, k:%s' % (args.method, args.dataset, args.num_seed, args.c, args.k)
		else:
			prt_str = '%s, %s, c:%s, k:%s' % (args.method, args.dataset, args.c, args.k)
		print(prt_str)
	elif args.method == 'RWR_imp':
		if args.is_sample:
			prt_str = '%s, %s, #seedu:%d, c:%s, alpha:%s, k:%s' % (args.method, args.dataset, args.num_seed, args.c, args.alpha, args.k)
		else:
			prt_str = '%s, %s, c:%s, alpha:%s, k:%s' % (args.method, args.dataset, args.c, args.alpha, args.k)
		print(prt_str)
	elif args.method == 'RWR_bias':
		if args.is_sample:
			prt_str = '%s, %s, #seedu:%d, c:%s, beta:%s, gamma:%s, k:%s' % (args.method, args.dataset, args.num_seed, args.c, args.beta, args.gamma, args.k)
		else:
			prt_str = '%s, %s, c:%s, beta:%s, gamma:%s, k:%s' % (args.method, args.dataset, args.c, args.beta, args.gamma, args.k)
		print(prt_str)
	elif args.method == 'RWR_side':
		if args.is_sample:
			prt_str = '%s, %s, #seedu:%d, c:%s, delta:%s, k:%s' % (args.method, args.dataset, args.num_seed, args.c, args.delta, args.k)
		else:
			prt_str = '%s, %s, c:%s, delta:%s, k:%s' % (args.method, args.dataset, args.c, args.delta, args.k)
		print(prt_str)


if __name__ == '__main__':
	main()
