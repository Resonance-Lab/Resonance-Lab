# -*- coding: utf-8 -*-
"""

@Project  :  Resonance Quantitative Research Lab —— 因子算子
@Author   :  Songhuai
@Version  :  1.0.0
@Contact  :  1155238565@link.cuhk.edu.hk
@Location :  Hong Kong, China
@Date     :  2025-11-14

"""

import numpy as np
import pandas as pd

def skew_vol_resonance_factor(returns, close, volume, oi,
                              skew_window_short=20, skew_window_long=60):
    """
    偏度-成交量共振因子
    """
    # 计算收益率
    returns_df = close.pct_change()
    
    # 滚动偏度
    skew_short = returns_df.rolling(window=skew_window_short).skew()
    skew_long = returns_df.rolling(window=skew_window_long).skew()
    
    # 偏度变化
    skew_change = (skew_short - skew_long) / (skew_long.abs() + 0.01)
    
    # 偏度分位数
    skew_percentile = skew_short.rolling(120).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1] * 100) if len(x) > 0 else 50
    )
    
    # 成交量状态
    vol_ma = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    
    vol_score = pd.DataFrame(1.0, index=volume.index, columns=volume.columns)
    vol_score[volume > vol_ma + 1.5 * vol_std] = 1.5
    vol_score[(volume > vol_ma + 0.5 * vol_std) & 
              (volume <= vol_ma + 1.5 * vol_std)] = 1.2
    vol_score[volume < vol_ma - 0.5 * vol_std] = 0.8
    
    # 价格动量
    return_5d = close.pct_change(5)
    
    # 信号识别
    signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    
    condition1 = (skew_change > 0.3) & (skew_short > -0.5) & \
                 (vol_score > 1.2) & (return_5d > 0)
    signal[condition1] = 2
    
    condition2 = (skew_change < -0.3) & (skew_short < 0.5) & \
                 (vol_score > 1.2) & (return_5d < 0)
    signal[condition2] = -2
    
    condition3 = (skew_percentile.sub(50).abs() > 40) & (vol_score < 1.0)
    signal[condition3] = -np.sign(skew_short[condition3])
    
    condition_other = ~(condition1 | condition2 | condition3)
    signal[condition_other] = 0.5 * np.sign(skew_change[condition_other]) * \
                              vol_score[condition_other]
    
    # 动量确认
    atr = returns_df.abs().rolling(5).mean()
    momentum_strength = return_5d.abs() / (atr + 1e-8)
    
    # 综合因子
    factor = signal * (1 + 0.5 * momentum_strength) * vol_score
    
    return factor