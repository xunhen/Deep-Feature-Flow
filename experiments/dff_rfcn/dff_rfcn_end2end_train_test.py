# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import os
import sys

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'dff_rfcn'))
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'dff_rfcn','core'))
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'dff_rfcn','operate_cxx'))
sys.path.append(os.path.join(this_dir, '..', '..'))
sys.path.append(os.path.join(this_dir, '..', '..', 'lib'))
import train_end2end
import train_eval_end2end
import test

if __name__ == "__main__":
    # train_end2end.main()
    test.main()
