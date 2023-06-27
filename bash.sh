PyMIC_path=/home/x/projects/PyMIC_project/PyMIC_release/PyMIC-master
export PYTHONPATH=$PYTHONPATH:$PyMIC_path

# python train.py demo/pctnet_scratch.cfg
# python predict.py demo/pctnet_scratch.cfg

python train.py demo/pctnet_pretrain.cfg
# python predict.py demo/pctnet_scratch.cfg

# python $PyMIC_path/pymic/util/evaluation_seg.py -cfg demo/evaluation.cfg
