# -*- coding: utf-8 -*-
import skimage.io

from transfer.transfer_all import TransferAll

if '__main__' == __name__:
    image_original = skimage.img_as_float(skimage.io.imread('sample/image/1.jpg'))
    image_sample = skimage.img_as_float(skimage.io.imread('sample/good/good-001.jpg'))
    transfer = TransferAll()
    image_result = transfer.transfer(image_original=image_original, image_sample=image_sample)
    skimage.io.imsave('result/result.jpg', image_result)
