# -*- coding: utf-8 -*-

# 第三方依赖产生的噪音警告会淹没进度条，统一压制（不影响本工具的异常处理）
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="insightface")
