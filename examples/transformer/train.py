# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.io import DataLoader

from utils.configure import PDConfig
from utils.check import check_gpu, check_version

from hapi.model import Input, set_device
from hapi.callbacks import ProgBarLogger
from reader import create_data_loader
from transformer import Transformer, CrossEntropyCriterion


class TrainCallback(ProgBarLogger):
    def __init__(self,
                 args,
                 verbose=2,
                 train_steps_fn=None,
                 eval_steps_fn=None):
        # TODO(guosheng): save according to step
        super(TrainCallback, self).__init__(args.print_step, verbose)
        # the best cross-entropy value with label smoothing
        loss_normalizer = -(
            (1. - args.label_smooth_eps) * np.log(
                (1. - args.label_smooth_eps)) + args.label_smooth_eps *
            np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))
        self.loss_normalizer = loss_normalizer
        self.train_steps_fn = train_steps_fn
        self.eval_steps_fn = eval_steps_fn

    def on_train_begin(self, logs=None):
        super(TrainCallback, self).on_train_begin(logs)
        self.train_metrics += ["normalized loss", "ppl"]

    def on_train_batch_begin(self, step, logs=None):
        if step == 0 and self.train_steps_fn:
            self.train_progbar._num = self.train_steps_fn()

    def on_train_batch_end(self, step, logs=None):
        logs["normalized loss"] = logs["loss"][0] - self.loss_normalizer
        logs["ppl"] = np.exp(min(logs["loss"][0], 100))
        super(TrainCallback, self).on_train_batch_end(step, logs)

    def on_eval_begin(self, logs=None):
        super(TrainCallback, self).on_eval_begin(logs)
        self.eval_metrics = list(
            self.eval_metrics) + ["normalized loss", "ppl"]

    def on_eval_batch_begin(self, step, logs=None):
        if step == 0 and self.eval_steps_fn:
            self.eval_progbar._num = self.eval_steps_fn()

    def on_eval_batch_end(self, step, logs=None):
        logs["normalized loss"] = logs["loss"][0] - self.loss_normalizer
        logs["ppl"] = np.exp(min(logs["loss"][0], 100))
        super(TrainCallback, self).on_eval_batch_end(step, logs)


def do_train(args):
    device = set_device("gpu" if args.use_cuda else "cpu")
    fluid.enable_dygraph(device) if args.eager_run else None

    # set seed for CE
    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        fluid.default_main_program().random_seed = random_seed
        fluid.default_startup_program().random_seed = random_seed

    # define inputs
    inputs = [
        Input(
            [None, None], "int64", name="src_word"),
        Input(
            [None, None], "int64", name="src_pos"),
        Input(
            [None, args.n_head, None, None],
            "float32",
            name="src_slf_attn_bias"),
        Input(
            [None, None], "int64", name="trg_word"),
        Input(
            [None, None], "int64", name="trg_pos"),
        Input(
            [None, args.n_head, None, None],
            "float32",
            name="trg_slf_attn_bias"),
        Input(
            [None, args.n_head, None, None],
            "float32",
            name="trg_src_attn_bias"),
    ]
    labels = [
        Input(
            [None, 1], "int64", name="label"),
        Input(
            [None, 1], "float32", name="weight"),
    ]

    # def dataloader
    (train_loader, train_steps_fn), (
        eval_loader, eval_steps_fn) = create_data_loader(args, device)

    # define model
    transformer = Transformer(
        args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
        args.n_layer, args.n_head, args.d_key, args.d_value, args.d_model,
        args.d_inner_hid, args.prepostprocess_dropout, args.attention_dropout,
        args.relu_dropout, args.preprocess_cmd, args.postprocess_cmd,
        args.weight_sharing, args.bos_idx, args.eos_idx)

    transformer.prepare(
        fluid.optimizer.Adam(
            learning_rate=fluid.layers.noam_decay(
                args.d_model,
                args.warmup_steps,
                learning_rate=args.learning_rate),
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=float(args.eps),
            parameter_list=transformer.parameters()),
        CrossEntropyCriterion(args.label_smooth_eps),
        inputs=inputs,
        labels=labels,
        device=device)

    ## init from some checkpoint, to resume the previous training
    if args.init_from_checkpoint:
        transformer.load(args.init_from_checkpoint)
    ## init from some pretrain models, to better solve the current task
    if args.init_from_pretrain_model:
        transformer.load(args.init_from_pretrain_model, reset_optimizer=True)

    # model train
    transformer.fit(train_data=train_loader,
                    eval_data=eval_loader,
                    epochs=args.epoch,
                    eval_freq=1,
                    save_freq=1,
                    save_dir=args.save_model,
                    callbacks=[
                        TrainCallback(
                            args,
                            train_steps_fn=train_steps_fn,
                            eval_steps_fn=eval_steps_fn)
                    ])


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()
    check_gpu(args.use_cuda)
    check_version()

    do_train(args)
