cd "./src/code/"
echo "Split the datasets into 5 folds"
python main.py --method "split5folds"
echo "MF_exp with filmtrust dataset"
python main.py --config True --config_path ../config/democonfig.conf
