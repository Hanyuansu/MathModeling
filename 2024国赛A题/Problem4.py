import math
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# --------------- 基本参数 ---------------
PITCH = 0.55                        # 螺距 p（m）
B = PITCH / (2.0 * math.pi)         # b = p/(2π)
THETA0 = 2.0 * math.pi * 16         # t=0 时龙头角（第 16 圈 A 点）
V = 1.0                             # 龙头弧长速度（m/s）