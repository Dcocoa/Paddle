#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import argparse
import os
import sys
import logging
import paddle.fluid as fluid

from hapi.model import set_device, Input

from modeling import bmn, BmnLoss
from bmn_metric import BmnMetric
from reader import BmnDataset
from config_utils import *

DATATYPE = 'float32'

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("BMN test for performance evaluation.")
    parser.add_argument(
        "-d",
        "--dynamic",
        action='store_true',
        help="enable dygraph mode, only support dynamic mode at present time")
    parser.add_argument(
        '--config_file',
        type=str,
        default='bmn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--device',
        type=str,
        default='gpu',
        help='gpu or cpu, default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path. None to automatically download weights provided by Paddle.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='output dir path. None to use config file setting.')
    parser.add_argument(
        '--result_path',
        type=str,
        default=None,
        help='output dir path after post processing. None to use config file setting.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


# Performance Evaluation
def test_bmn(args):
    device = set_device(args.device)
    fluid.enable_dygraph(device) if args.dynamic else None

    #config setting
    config = parse_config(args.config_file)
    eval_cfg = merge_configs(config, 'test', vars(args))

    feat_dim = config.MODEL.feat_dim
    tscale = config.MODEL.tscale
    dscale = config.MODEL.dscale
    prop_boundary_ratio = config.MODEL.prop_boundary_ratio
    num_sample = config.MODEL.num_sample
    num_sample_perbin = config.MODEL.num_sample_perbin

    #input and video index
    inputs = [
        Input(
            [None, config.MODEL.feat_dim, config.MODEL.tscale],
            'float32',
            name='feat_input')
    ]
    gt_iou_map = Input(
        [None, config.MODEL.dscale, config.MODEL.tscale],
        'float32',
        name='gt_iou_map')
    gt_start = Input([None, config.MODEL.tscale], 'float32', name='gt_start')
    gt_end = Input([None, config.MODEL.tscale], 'float32', name='gt_end')
    video_idx = Input([None, 1], 'int64', name='video_idx')
    labels = [gt_iou_map, gt_start, gt_end, video_idx]

    #data
    eval_dataset = BmnDataset(eval_cfg, 'test')

    #model
    model = bmn(tscale,
                dscale,
                prop_boundary_ratio,
                num_sample,
                num_sample_perbin,
                pretrained=args.weights is None)
    model.prepare(
        loss_function=BmnLoss(tscale, dscale),
        metrics=BmnMetric(
            config, mode='test'),
        inputs=inputs,
        labels=labels,
        device=device)

    #load checkpoint
    if args.weights is not None:
        assert os.path.exists(args.weights + '.pdparams'), \
            "Given weight dir {} not exist.".format(args.weights)
        logger.info('load test weights from {}'.format(args.weights))
        model.load(args.weights)

    model.evaluate(
        eval_data=eval_dataset,
        batch_size=eval_cfg.TEST.batch_size,
        num_workers=eval_cfg.TEST.num_workers,
        log_freq=args.log_interval)

    logger.info("[EVAL] eval finished")


if __name__ == '__main__':
    args = parse_args()
    test_bmn(args)
