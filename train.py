# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import os
import sys
import shutil
from datetime import datetime
from pymic.util.parse_config import *
from pymic.net_run.agent_seg import SegmentationAgent
from net.pct_net import PCTNet

def main():
    """
    The main function for running a network for training.
    """
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('   python train.py config.cfg')
        exit()
    cfg_file = str(sys.argv[1])
    if(not os.path.isfile(cfg_file)):
        raise ValueError("The config file does not exist: " + cfg_file)
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    log_dir  = config['training']['ckpt_save_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)
    dst_cfg = cfg_file if "/" not in cfg_file else cfg_file.split("/")[-1]
    shutil.copy(cfg_file, log_dir + "/" + dst_cfg)
    datetime_str = str(datetime.now())[:-7].replace(":", "_")
    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_train_{0:}.txt".format(datetime_str), 
                            level=logging.INFO, format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_train_{0:}.txt".format(datetime_str), 
                            level=logging.INFO, format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)

    agent = SegmentationAgent(config, 'train') 
    mynet  = PCTNet(config['network'])
    agent.set_network(mynet)
    agent.run()

if __name__ == "__main__":
    main()
    

