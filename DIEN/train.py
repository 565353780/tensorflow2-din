#!/usr/bin/env python
# -*- coding: utf-8 -*-

from DINTrainer import DINTrainer

'''
method:
    0: "Source"
    1: "AFM-Add-to-Output" # make output wrong size (32, 32)
    2: "AFM-Add-to-Attention-Output"
    3: "AFM-With-Candidate"
pos_list_len_max:
    >1: set pos_list in dataset not longer than this value
    <1: pos_list length no limit
use_din_source_method:
    True: use din source method to create dataset
    False: use new method in din-tf2 to create dataset
'''

if __name__ == '__main__':
    din_trainer = DINTrainer()
    din_trainer.init_env(method_idx=3, pos_list_len_max=100, use_din_source_method=True)
    din_trainer.train()

