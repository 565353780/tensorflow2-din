#!/usr/bin/env python
# -*- coding: utf-8 -*-

from DINTrainer import DINTrainer

'''
# method:
    0: "Source"
    1: "AFM-Add-to-Output"
    2: "AFM-Add-to-Attention-Output"
    3: "AFM-With-Candidate"
'''

if __name__ == '__main__':
    din_trainer = DINTrainer()
    din_trainer.init_env(method_idx=3)
    din_trainer.train()

