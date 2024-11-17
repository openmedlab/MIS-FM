#PyMIC_path=/home/x/projects/PyMIC_project/PyMIC_release/PyMIC-master
#PyMIC_path=/home/disk4t/projects/PyMIC_project/PyMIC
#export PYTHONPATH=$PYTHONPATH:$PyMIC_path

python net_run.py train demo/fmunet_pretrain.cfg
# python net_run.py test  demo/fmunet_pretrain.cfg

# python net_run.py train demo/fmunet_scratch.cfg
# python net_run.py test  demo/fmunet_scratch.cfg

# python net_run.py train demo/pctnet_pretrain.cfg
# python net_run.py test  demo/pctnet_pretrain.cfg

# python net_run.py train demo/pctnet_scratch.cfg
# python net_run.py test  demo/pctnet_scratch.cfg

# pymic_eval_seg -cfg demo/evaluation.cfg


