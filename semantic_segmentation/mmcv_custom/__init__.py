# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# -*- coding: utf-8 -*-

# from .apex_runner.optimizer import DistOptimizerHook
from .checkpoint import load_checkpoint
from .customized_text import CustomizedTextLoggerHook
from .layer_decay_optimizer_constructor import \
    LearningRateDecayOptimizerConstructor
from .resize_transform import SETR_Resize

# from .train_api import train_segmentor

# __all__ = ['load_checkpoint', 'LearningRateDecayOptimizerConstructor', 'SETR_Resize', 'DistOptimizerHook', 'train_segmentor', 'CustomizedTextLoggerHook']
__all__ = ['load_checkpoint', 'LearningRateDecayOptimizerConstructor', 'SETR_Resize', 'CustomizedTextLoggerHook']
