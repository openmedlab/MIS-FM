# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import logging
import os
import sys
import shutil
from datetime import datetime
from pymic.util.parse_config import *
from pymic.net.net_dict_seg import SegNetDict
from pymic.net_run.agent_seg import SegmentationAgent
from net.pct_net import PCTNet
from net.fmunet import FMUNet

net_dict = {'PCTNet':PCTNet, 
            'FMUNet':FMUNet}
net_dict.update(SegNetDict)

def main():
    """
    The main function for running a network for training.
    """
    if(len(sys.argv) < 3):
        print('Number of arguments should be at least 3. e.g.')
        print('   python net_run.py train config.cfg')
        print('   python net_run.py test  config.cfg')
        exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", help="train or test stage")
    parser.add_argument("cfg", help="configuration file for training/testing")
    args = parser.parse_args()
    assert(args.stage in ("train", "test"))
    if(not os.path.isfile(args.cfg)):
        raise ValueError("The config file does not exist: " + args.cfg)
    config   = parse_config(args)
    config   = synchronize_config(config)
    config["network"]["multiscale_pred"] = False
    
    if args.stage == "train":
        log_dir  = config['training']['ckpt_dir']
    else:
        log_dir  = config['testing']['output_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)
    dst_cfg = args.cfg if "/" not in args.cfg else args.cfg.split("/")[-1]
    shutil.copy(args.cfg, log_dir + "/" + dst_cfg)
    datetime_str = str(datetime.now())[:-7].replace(":", "_")
    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_{0:}_{1:}.txt".format(args.stage, datetime_str), 
                            level=logging.INFO, format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_{0:}_{1:}.txt".format(args.stage, datetime_str), 
                            level=logging.INFO, format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)

    agent = SegmentationAgent(config, args.stage) 
    agent.set_net_dict(net_dict)
    agent.run()

if __name__ == "__main__":
    main()
    

