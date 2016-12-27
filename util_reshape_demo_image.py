# -*- coding: utf-8 -*-

import os

import skimage.io
import skimage.transform

import config

def process(base_dir):
    for parent, dirnames, filenames in os.walk(base_dir):
        n_images = len(filenames)
        for ith in range(n_images):
            filename = filenames[ith]
            print(ith + 1, '/', n_images, ' ', filename)
            try:
                # 读取图片
                image = skimage.img_as_float(skimage.io.imread(os.path.join(base_dir, filename)))
                height, width = image.shape[0], image.shape[1]
                if height > width:
                    width = int(width / height * 800)
                    height = 800
                else:
                    height = int(height / width * 800)
                    width = 800

                image = skimage.transform.resize(image, (height, width))
                skimage.io.imsave(os.path.join(base_dir, filename), image)
            except Exception as e:
                print("Error:", filename)


if '__main__' == __name__:
    process(config.PATH_CRAW)
