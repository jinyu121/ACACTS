# -*- coding: utf-8 -*-

import os

# 路径
PATH_OPENCV_FACE = '/home/haoyu/ProgramFiles/opencv3/data/haarcascades/haarcascade_frontalface_default.xml'
PATH_CAFFE = '/home/haoyu/ProgramFiles/nlpcaffe'
PATH_IMAGE_GOOD = 'sample/good'
PATH_IMAGE_BAD = 'sample/bad'
PATH_CRAW = 'craw'

# 开关
USE_LUMINANCE_CORRECT = False
USE_PCA = False
USE_REDIS = False

# 配置
crawler = {
    'api_key': '5206ff5fde8941b99ce03d1f36bf1397',
    'pages_per_day': 1,
    'savers': 5
}

search = {
    'top_k': 3
}

transfer_pre = {
    'gamma': 2.2,
    'drop_rate': 0.005,
}

transfer_color = {
    'lambda_r': 7.5,
}

transfer_luminance = {
    'tau': 0.4,
    'num_of_samples': 32,
}

transfer_face = {
    'luminance_th_r': 0.5,
    'gamma_th': 0.5,
    'alpha_r': 0.45,
    'alpha_c': 0.001,
}

caffe = {
    'root': PATH_CAFFE,
    'root_python': os.path.join(PATH_CAFFE, 'python'),
    'file_deploy_proto': os.path.join(PATH_CAFFE, 'models/bvlc_alexnet/deploy.prototxt'),
    'file_caffe_model': os.path.join(PATH_CAFFE, 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'),
}

kmeans = {
    'n_clusters': 1000,
}

style_ranking = {
    'lambda_l': 0.005,
    'lambda_c': 0.05,
    'epsilon': 1,
    'scaling': (600, 800),
}
