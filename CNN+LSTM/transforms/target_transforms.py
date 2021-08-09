import math
import random


# 组合转换函数
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst


# 获取类别
class ClassLabel(object):

    def __call__(self, target):
        return target['label']


# 获取视频id
class VideoID(object):

    def __call__(self, target):
        return target['video_id']