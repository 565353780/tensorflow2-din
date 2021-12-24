#!/usr/bin/env python
# -*- coding: utf-8 -*-

from DINTrainer import DINTrainer

'''
method:
DIN:
    0: "Source"
    1: "AFM-Add-to-Output" # make output wrong size (32, 32)
    2: "AFM-Add-to-Attention-Output"
    3: "AFM-With-Candidate"
Attention:
    0: "Source"
    1: "Add-Conv2D-to-Attention"

pos_list_len_max:
    >1: set pos_list in dataset not longer than this value
    <1: pos_list length no limit

use_din_source_method:
    True: use din source method to create dataset
    False: use new method in din-tf2 to create dataset

source_lr: source learning rate

decay_rate: learning rate will multiply to this after every decay_steps
    if it is set to None, it will equal to print_step

decay_steps: learning rate will be changed after steps with this value
    if it is set to None, it will equal to dataset's size as default
    better to equal to dataset's size
'''

if __name__ == '__main__':
    din_trainer = DINTrainer()
    din_trainer.init_env(method_idx=[0, 1],
                         pos_list_len_max=60,
                         use_din_source_method=True,
                         source_lr=0.1,
                         decay_rate=0.99,
                         decay_steps=None)
    din_trainer.train()

