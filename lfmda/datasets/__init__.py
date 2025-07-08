'''
@Project : rads2 
@File    : __init__.py.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2023/5/26 下午4:52
@e-mail  : 1183862787@qq.com
'''
from .loveda import LoveDA
from .isprsda import IsprsDA
from .robotic import Robotic
from .aeroscapes import Aeroscapes
from .uavid2020 import UAVid2020
from .daLoader import DALoader


__all__ = ['LoveDA', 'IsprsDA', 'DALoader',
           'Aeroscapes',
           'UAVid2020',
           'Robotic']
