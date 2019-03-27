# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import logging
import time

import mxnet as mx


class Speedometer(object):
    def __init__(self, batch_size, frequent=50, sw=None):
        self.batch_size = batch_size
        self.frequent = frequent
        self.sw = sw
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                s = ''
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-" % (param.epoch, count, speed)
                    if self.sw:
                        self.sw.add_scalar('epoch{}/speed'.format(param.epoch), speed, global_step=param.nbatch)
                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                else:
                    s = "Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (param.epoch, count, speed)

                logging.info(s)
                print(s)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


class SummaryMetric(object):
    def __init__(self, sw, frequent=50, prefix=None):
        self.frequent = frequent
        self.sw = sw
        self.prefix = '' if not prefix else prefix + '/'

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if count % self.frequent == 0:
            if param.eval_metric is not None:
                name, value = param.eval_metric.get()
                for n, v in zip(name, value):
                    self.sw.add_scalar(self.prefix + "epoch{}/".format(param.epoch) + n, v, global_step=count)

#need change!!!
class SummaryValMetric(object):
    def __init__(self, sw, prefix=None):
        self.sw = sw
        self.prefix = '' if not prefix else prefix + '/'

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.epoch
        if param.eval_metric is not None:
            name, value = param.eval_metric.get()
            for n, v in zip(name, value):
                self.sw.add_scalar(self.prefix + n, v, global_step=count)


def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        weight = arg['rfcn_bbox_weight']
        bias = arg['rfcn_bbox_bias']
        repeat = bias.shape[0] / means.shape[0]

        arg['rfcn_bbox_weight_test'] = weight * mx.nd.repeat(mx.nd.array(stds), repeats=int(repeat)).reshape(
            (bias.shape[0], 1, 1, 1))
        arg['rfcn_bbox_bias_test'] = arg['rfcn_bbox_bias'] * mx.nd.repeat(mx.nd.array(stds),
                                                                          repeats=int(repeat)) + mx.nd.repeat(
            mx.nd.array(means), repeats=int(repeat))
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop('rfcn_bbox_weight_test')
        arg.pop('rfcn_bbox_bias_test')

    return _callback
