#!/usr/bin/env python
# -*- coding: utf-8 -*-

from DINTrainer import DINTrainer

if __name__ == '__main__':
    din_trainer = DINTrainer()
    din_trainer.init_env(method_idx=3)
    din_trainer.train()

