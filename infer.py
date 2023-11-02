#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import init_path
import core.config as config
import core.distributed as dist
import core.trainer as trainer
from core.config import cfg


def main():
    config.load_cfg_fom_args("Infer a CFCD model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.infer_model)


if __name__ == "__main__":
    main()
