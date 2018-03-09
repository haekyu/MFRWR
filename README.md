
# Readme of mfrwr codes (v1.0)


## Contents

1. Basic information
2. Overview
3. Requirements
4. How to install
5. How to use
	1) Input and output
	2) How to run
	3) How to give parameters
6. Demo example

## 1. Basic information

- Authors: Haekyu Park, Jinhong Jung, and U Kang
- Program name: MFRWR
- Version: 1.0
- Last updated: 28 Aug 2017
- Main contack: Haekyu Park (hkpark627@snu.ac.kr)

## 2. Overview

This package is a set of implementaions of recommender systems based on matrix factorization and random walk with restart.
These methods are compared in each recommendation scenarios in the following paper: [**A comparative study of matrix factorization and random walk with restart in recommender system.**](https://datalab.snu.ac.kr/mfrwr/resources/mfrwr.pdf)
We suggest 4 matrix factorization methods and 4 random walk with restart methods for the following cases:
- When explicit feedback ratings are given
- When implicit feedback ratings are given
- When bias terms are introduced
- When side information is used

## 3. Requirements

- python 3.*
- numpy
- pandas
- scipy
- We recommend you to use [Anaconda](https://www.continuum.io/downloads).

## 4. How to install
You can download this code package at the [project hompage](https://datalab.snu.ac.kr/mfrwr).


## 5. How to use
1) Input and output
- Input
    - Ratings and side information are able to be given as input.
    - All the input should be given as tab-separated files.
        - Rating file should have three columns for user_id, item_id, and rating.
        - User_id should not overlapped with item_id.
        - Side information file should have two columns for user/item_id and value.
    - The name of rating files should be 'rating.tsv'.
- Output
    - We print out Spearman's rho, precision@k, and recall@k with stdout.
    - You may write log generator for the results for yourself.
- Intermediate outputs
    - We generate intermediate outputs such as vectors and biases of users and items.
    - All the intermediate outputs have their file names under the rule: 'method name_dataset name_parameters_fold.txt'.
        
2) How to run
- First go to `./code/`.
- You can run the code by typing `python main.py`.
- You can optionally give parameters with two approaches.
	* 1) By appending '--argument_type argument_value'.
	For example, if you want to run matrix factorization with explicit ratings, and you want to set learning rate = 0.05, lambda = 0.3, and dimension = 5, 
	you can run the code as follows:
	python main.py --method MF_exp --lr 0.05 --lamb 0.3 --dim 5

	* 2) By using config
	You can make a configuration file that contains all the parameters you give.
	Then you can run the code by giving path of the config file as follows: 
	python main.py --config True --config_path ./myconfig.conf
	Rules to make config files are as follows.
		- All parameters should be separated with '\n'.
		- Argument type and its values for each parameters should be separated with ';'.
	For example, a config file to run matrix factorization is as follows.
	```
	method;MF_exp
	dataset;filmtrust
	lr;0.05
	lamb;0.3
	dim;5
	```

The parameters you can give are as follows.

| argument_type	|	default argument_value		|	details       			|
|---| ---| ---|
|--dataset 	| filmtrust				| Name of dataset			|
|--method 	| MF_exp				| Name of method (*1)			|
|--data_path	| '../data/'				| Where datasets are			|
|--input_path	| '../data/<dataset>/input/'		| Where inputs are			|
|--result_path	| '../results'				| Where intermediate results are	|
|--side_paths	| [<input_path>/link.tsv]		| List of paths of side info (*2)	|
|--entity_types	| [['u', 'u']]				| List of types of side info (*3)	|
|--config 	| False					| Whether to give config file		|
|--config_path	| '../config/democonfig.conf'		| Path of config file			|
|--is_implicit	| False					| Whether implicit data are given	|
|--alpha	| 0.001					| Coefficient of confidence 		|
|		|					|    level in implicit feedback		|
|--is_social	| False					| Whether social links are used		|
|--lr 		| 0.05					| Learning rate				|
|--lamb 	| 0.3					| Regularization parameters		|
|--dim 		| 5					| Dimension of vectors			|
|--is_sample	| False					| Wheter to sample seed users (*4) 	|
|--num_seed	| 300					| # sampled seed users			|
|--c 		| 0.2					| Probability of restart		|
|--beta		| 0.4					| Probability of walk in RWR_bias	|
|--gamma	| 0.3					| Probability of restart in RWR_bias	|
|--delta	| 1.0					| Weight of additional links in RWR_side| 

  

- (*1) 
    - Name of methods can be one of the followings.
	- split5folds, MF_exp, MF_imp, MF_bias, MF_side, RWR_exp, RWR_imp, RWR_bias, RWR_side.
	- When 'split5folds' is given, you can split rating data into 5 folds.
	The split should be done before running MF/RWR methods.

- (*2)
    - You can give file paths of side information with list.
	- This should be given by config file.
	- For example, if you want to give '../data/movielens/age.tsv' and '../data/movielens/gender.tsv' for side information, you can give --side_paths argument in the config file as follows.
	```
	side_paths;['../data/movielens/age.tsv', '../data/movielens/gender.tsv']
	```

- (*3)
    - You should declare types of entities in side information.
	- Each type can be one of the followings: 'u', 'i', and 's'.
	- 'u' indicates users, 'i' indicates items, and 's' indicates similarity attributes.
	- This should be given by config file.
	- For example, if you have '../data/movielens/age.tsv' and '../data/movielens/gender.tsv' which have first column for user id and second column for user attribute, a config file can be written as follows.
	```
	side_paths;['../data/movielens/age.tsv', '../data/movielens/gender.tsv']
	entity_types;[['u', 's'], ['u', 's']].
	```

- (*4)
    - You can sample seed users for RWR methods.
	- Sampling options are given because the methods take too much time if many users are included.



## 6. Demo example
You can run MF_exp with filmtrust dataset.
Please run demo.sh by typing `./demo.sh`.
