# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import skimage

import config as ac_conf
from feature.feature_base import FeatureBase

sys.path.insert(0, ac_conf.caffe['root_python'])
import caffe


class FeatureDeep(FeatureBase):
    # 设置GPU模式
    # caffe.set_mode_gpu()
    # 均值文件
    __mu = np.load(os.path.join(ac_conf.caffe['root_python'], 'caffe/imagenet/ilsvrc_2012_mean.npy'))
    __mu = __mu.mean(1).mean(1)
    # 加载model和network
    __net = caffe.Net(ac_conf.caffe['file_deploy_proto'], ac_conf.caffe['file_caffe_model'], caffe.TEST)
    # 设定图片的shape格式
    __transformer = caffe.io.Transformer({'data': __net.blobs['data'].data.shape})
    # 改变维度的顺序
    __transformer.set_transpose('data', (2, 0, 1))
    # 设置均值
    __transformer.set_mean('data', __mu)
    # 颜色归一化到[0,255]
    __transformer.set_raw_scale('data', 255)
    # 颜色变成BGR
    __transformer.set_channel_swap('data', (2, 1, 0))
    __net.blobs['data'].reshape(1,  # batch size
                                3,  # 3-channel (BGR) images
                                227, 227)  # image size is 227x227

    def extract(self, image):
        # 使用灰度加载图片
        im = skimage.img_as_float(image).astype(np.float32)
        # 执行上面设置的图片预处理操作，并将图片载入到blob中
        self.__net.blobs['data'].data[...] = self.__transformer.preprocess('data', im)
        # 执行测试
        out = self.__net.forward()
        # 提取某层数据（特征）
        feature = self.__net.blobs['fc6'].data.ravel()
        return feature
