#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import io
import tarfile
import numpy as np
from PIL import Image

from paddle.io import Dataset
from paddle.dataset.common import _check_exists_and_download

__all__ = ["VOC2012"]

VOC_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/\
VOCtrainval_11-May-2012.tar'

VOC_MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
SET_FILE = 'VOCdevkit/VOC2012/ImageSets/Segmentation/{}.txt'
DATA_FILE = 'VOCdevkit/VOC2012/JPEGImages/{}.jpg'
LABEL_FILE = 'VOCdevkit/VOC2012/SegmentationClass/{}.png'

CACHE_DIR = 'voc2012'

MODE_FLAG_MAP = {'train': 'trainval', 'test': 'train', 'valid': "val"}


class VOC2012(Dataset):
    """
    Implementation of `VOC2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>`_ dataset

    Args:
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train', 'valid' or 'test' mode. Default 'train'.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Examples:

        .. code-block:: python

            import paddle
            from paddle.incubate.hapi.datasets import VOC2012

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()

                def forward(self, image, label):
                    return paddle.sum(image), label

            paddle.disable_static()

            voc2012 = VOC2012(mode='train')

            for i in range(10):
                image, label= voc2012[i]
                image = paddle.cast(paddle.to_tensor(image), 'float32')
                label = paddle.to_tensor(label)

                model = SimpleNet()
                image, label= model(image, label)
                print(image.numpy().shape, label.numpy().shape)

    """

    def __init__(self,
                 data_file=None,
                 mode='train',
                 transform=None,
                 download=True):
        assert mode.lower() in ['train', 'valid', 'test'], \
            "mode should be 'train', 'valid' or 'test', but got {}".format(mode)
        self.flag = MODE_FLAG_MAP[mode.lower()]

        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file is not set and downloading automatically is disabled"
            self.data_file = _check_exists_and_download(
                data_file, VOC_URL, VOC_MD5, CACHE_DIR, download)
        self.transform = transform

        # read dataset into memory
        self._load_anno()

    def _load_anno(self):
        self.name2mem = {}
        self.data_tar = tarfile.open(self.data_file)
        for ele in self.data_tar.getmembers():
            self.name2mem[ele.name] = ele

        set_file = SET_FILE.format(self.flag)
        sets = self.data_tar.extractfile(self.name2mem[set_file])

        self.data = []
        self.labels = []

        for line in sets:
            line = line.strip()
            data = DATA_FILE.format(line.decode('utf-8'))
            label = LABEL_FILE.format(line.decode('utf-8'))
            self.data.append(data)
            self.labels.append(label)

    def __getitem__(self, idx):
        data_file = self.data[idx]
        label_file = self.labels[idx]

        data = self.data_tar.extractfile(self.name2mem[data_file]).read()
        label = self.data_tar.extractfile(self.name2mem[label_file]).read()
        data = Image.open(io.BytesIO(data))
        label = Image.open(io.BytesIO(label))
        data = np.array(data)
        label = np.array(label)
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.data)

    def __del__(self):
        if self.data_tar:
            self.data_tar.close()
