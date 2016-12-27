# -*- coding: utf-8 -*-

class TransferBase:
    def transfer(self, image_original=None, image_sample=None):
        """
        如果是简单处理，直接返回结果
        如果是复杂处理，需要通过 get_result 获取结果
        """
        pass

    def get_result(self):
        """
        用于复杂处理获取结果
        """
        pass
