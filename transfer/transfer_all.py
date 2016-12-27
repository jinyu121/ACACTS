# -*- coding: utf-8 -*-

from transfer.transfer_base import TransferBase
from transfer.transfer_style import TransferStyle


class TransferAll(TransferBase):
    def transfer(self, image_original=None, image_sample=None):
        if image_original is None or image_sample is None:
            raise Exception('必须是两张图片')

        # 做变换
        style_transfer = TransferStyle(image_original, image_sample)
        style_transfer.transfer()

        return style_transfer.get_result()
