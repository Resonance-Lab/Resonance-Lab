# -*- coding: utf-8 -*-
"""

@Project  :  Resonance Quantitative Research Lab —— 回测框架
@Author   :  Songhuai
@Version  :  1.0.0
@Contact  :  1155238565@link.cuhk.edu.hk
@Location :  Hong Kong, China
@Date     :  2025-11-13

"""

import os
import sys
import time
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import Union
from scipy.stats import skew, kurtosis

# 环境配置
os.environ['QHDATA_USER'] = 'user_lengzhen'
os.environ['QHDATA_PASSWORD'] = 'ebowcnN3WCBI'
sys.path.append('/home/mw/input/util6299/')
from base_tool import *
from qhdata.API import *
from qhdata.API.stock import *
from qhdata.API.future import get_dominant, get_fut_daily_by

warnings.filterwarnings('ignore')


# ==================== 配置类 ====================
class BacktestConfig:
    """
    回测配置类
    保留原始代码中的所有参数
    """
    def __init__(self):
        # 回测时间范围
        self.start_date = 20140101
        self.end_date = 20251103
        
        # 品种列表
        self.syms = ['a', 'ag', 'al', 'ap', 'au','ao', 'b', 'bb', 'bc', 'bu', 'c', 'cf',
                     'cj', 'cs', 'cu', 'eb', 'eg','ec', 'fb', 'fg', 'fu', 'hc', 'i',
                     'j', 'jd', 'jm', 'jr', 'l','lc', 'lh', 'lr', 'lu', 'm',
                     'ma', 'ni', 'nr', 'oi', 'p', 'pb', 'pf', 'pg', 'pk', 'pm', 'pp','pr','px', 'rb',
                     'ri', 'rm', 'rr', 'rs', 'ru', 'sa', 'sc', 'sf', 'si','sm', 'sn', 'sp', 'sr',
                     'ss','sh', 't', 'ta', 'tf', 'tl', 'ts', 'ur', 'v', 'wh', 'y', 'zc',
                     'zn','ps','im','if','ih','ic']
        self.varityLst = [i.lower() for i in self.syms]
        
        # 资金设置
        self.cashSetting = {"cash": 1e10, "cashLimit": 1e10*1.2, "targetAnnVola": 0.1}
        self.AUM = 1000000000000
        
        # 因子准备
        self.fields = ["Date", "Contract"]
        self.factor_direction = 1
        
        # IC计算参数
        self.ic_cal_method = 'pearson'
        self.ret_cal_win_size = 1
        self.rolling_t = 100
        
        # 信号生成参数
        self.n = 5  # 选收益前几的品种
        self.percent = 0.2  # 选择因子值前占比多少的品种
        self.bar = 0.8  # 选择因子在历史上占比前多少的品种
        self.r_len = 100  # 时序回测中回看多久
        
        # 交易参数
        self.tradeTime = 'nextOpen'
        self.costRate = 0.0005
        self.T = 1  # 持有几天
        self.T_hold = 1
        self.tc = 1
        
        # 文件路径
        self.file_address = './'
        self.f_name = None
        
        # TWAP参数
        self.twap_sapn = 5
        self.twap_part = 1
        
        # 其他
        self.right_off_flag = 'yes'
        self.score = False
        self.Max_min = False
        self.close_type = 'Settle'
        self.wtpy = False
        self.local = True
        
        # 费率设置
        self.fee = {"FUTURE": True}
        self.slip = {'XSHG': 0.00036, 'XSHE': 0.00036, 'CFFEX': 0.0002, 
                     'SHFE': 0.0002, 'DCE': 0.0002, 'CZCE': 0.0002, 'INE': 0.0002}
        
        # 窗口参数
        self.use_window = False  # 是否使用滚动窗口
        self.windows = [5, 10, 20, 60, 120]  # 滚动窗口列表


# ==================== 全局市场数据加载 ====================
def load_market_data(start_date, end_date, syms):
    """
    加载市场数据
    保留原始代码的数据加载逻辑
    """
    print(f"加载市场数据: {start_date} -> {end_date}")
    
    # 获取主力合约数据
    mkt = get_fut_daily_by(start_date, end_date, varity=syms, data_class='mc')
    varityLst = [i.lower() for i in syms]
    multi = all_instruments(market='future').groupby(['Varity']).Multi.last().loc[varityLst]
    
    # 数据预处理
    mkt = mkt.rename(columns={'Variety':'Symbol'}).set_index(['Date', 'Symbol']).sort_index()
    
    # 价格调整
    priceAdj = mkt.PriceAdj.unstack().cumsum()
    OpenAdj = mkt.Open.unstack() - priceAdj
    settlePriceAdj = mkt.SettlePrice.unstack() - priceAdj
    HighAdj = mkt.High.unstack() - priceAdj
    LowAdj = mkt.Low.unstack() - priceAdj
    CloseAdj = mkt.LastPrice.unstack() - priceAdj
    PreClosePriceAdj = mkt.PreClosePrice.unstack() - priceAdj
    PreSettlePricePriceAdj = mkt.PreSettlePrice.unstack() - priceAdj
    
    # 原始价格
    Open = mkt.Open.unstack()
    Settle = mkt.SettlePrice.unstack()
    High = mkt.High.unstack()
    Low = mkt.Low.unstack()
    Close = mkt.LastPrice.unstack()
    PreClose = mkt.PreClosePrice.unstack()
    PreSettlePrice = mkt.PreSettlePrice.unstack()
    
    # 成交量和持仓
    Volume = mkt.Volume.unstack()
    OI = mkt.OpenInterest.unstack()
    WeightedSettlePrice = mkt.WeightedSettlePrice.unstack()
    Turnover = mkt.Turnover.unstack()
    VolumeRatio = mkt.VolumeRatio.unstack()
    TurnoverRatio = mkt.TurnoverRatio.unstack()
    OIRatio = mkt.OIRatio.unstack()
    
    OIAll = OI / OIRatio
    VolumeAll = Volume / VolumeRatio
    TurnoverAll = Turnover / TurnoverRatio
    
    # 收益率
    ret_ctc = (CloseAdj - CloseAdj.shift(1)) / Close.shift(1)
    
    # 期限结构
    priceDf = get_fut_daily_by(start_date, end_date, data_class='ac')
    priceDf['Contract'] = priceDf['Contract'].str.lower()
    priceDf['Date'] = pd.to_datetime(priceDf['Date'], format='%Y%m%d')
    
    all_concract = get_dominant(start_date=start_date, end_date=end_date, rule=0, data_class='msnfc')
    all_concract.loc[all_concract['Near'] == '', 'Near'] = all_concract['Main']
    
    print("市场数据加载完成")
    
    # 返回所有数据
    return {
        'mkt': mkt,
        'multi': multi,
        'OpenAdj': OpenAdj,
        'settlePriceAdj': settlePriceAdj,
        'HighAdj': HighAdj,
        'LowAdj': LowAdj,
        'CloseAdj': CloseAdj,
        'PreClosePriceAdj': PreClosePriceAdj,
        'PreSettlePricePriceAdj': PreSettlePricePriceAdj,
        'Open': Open,
        'Settle': Settle,
        'High': High,
        'Low': Low,
        'Close': Close,
        'PreClose': PreClose,
        'PreSettlePrice': PreSettlePrice,
        'Volume': Volume,
        'OI': OI,
        'WeightedSettlePrice': WeightedSettlePrice,
        'Turnover': Turnover,
        'VolumeRatio': VolumeRatio,
        'TurnoverRatio': TurnoverRatio,
        'OIRatio': OIRatio,
        'OIAll': OIAll,
        'VolumeAll': VolumeAll,
        'TurnoverAll': TurnoverAll,
        'ret_ctc': ret_ctc,
        'priceDf': priceDf,
        'all_concract': all_concract
    }


# ==================== 标准化处理函数 ====================
def _check_type(signal):
    """检查数据类型"""
    return ('np', signal) if isinstance(signal, np.ndarray) else ('pd', signal.values)


def mad_clip(signal: Union[np.ndarray, pd.DataFrame], k: int = 5):
    """中位数绝对偏差去极值"""
    signal = signal.copy()
    dtype, signal_array = _check_type(signal)
    
    med = np.nanmedian(signal_array, axis=-1)[:, None]
    mad = np.nanmedian(np.abs(signal_array - med), axis=-1).reshape(-1, 1)
    signal_array = np.clip(signal_array, med - k * mad, med + k * mad)
    
    return signal_array if dtype == 'np' else pd.DataFrame(signal_array, index=signal.index, columns=signal.columns)


def std_clip(signal: pd.DataFrame, k: int = 3):
    """标准差去极值"""
    mean = signal.mean()
    stdev = signal.std()
    return signal.clip(mean - k * stdev, mean + k * stdev)


def min_max_norm(signal: Union[np.ndarray, pd.DataFrame]):
    """归一化"""
    signal = signal.copy()
    dtype, signal_array = _check_type(signal)
    
    min_ = np.nanmin(signal_array, axis=-1)[:, None]
    max_ = np.nanmax(signal_array, axis=-1)[:, None]
    signal_array = (signal_array - min_) / (max_ - min_)
    
    return signal_array if dtype == 'np' else pd.DataFrame(signal_array, index=signal.index, columns=signal.columns)


def std_norm(signal: Union[np.ndarray, pd.DataFrame]):
    """标准化"""
    signal = signal.copy()
    dtype, signal_array = _check_type(signal)
    
    mean = signal.mean()
    stdev = signal.std()
    signal_norm = (signal - mean) / (stdev + 1e-8)
    
    return signal_array if dtype == 'np' else pd.DataFrame(signal_array, index=signal.index, columns=signal.columns)


def rolling_vol_settle(f, k=250):
    """时序波动率处理"""
    def cal_vol(rt, n):
        return np.sqrt((rt ** 2).ewm(span=n, min_periods=50, adjust=False).mean()) * np.sqrt(244)
    
    f = f.loc[:, (f != 0).any(axis=0)]
    vol = f.dropna(how='all').apply(lambda col: cal_vol(col, k))
    f = (f / vol).dropna(how='all').dropna(axis=1, how='all').fillna(0).replace([np.inf, -np.inf], 0)
    
    return f


def standardlized(factor, k=3):
    """
    标准化处理
    1. 中位数绝对偏差去极值
    2. 时序波动率标准化
    3. 标准化
    """
    mad_clipped_df = mad_clip(factor, k=k)
    standardized_df = std_norm(rolling_vol_settle(mad_clipped_df))
    
    return standardized_df


# ==================== 索引转换函数 ====================
def set_index_from_int_to_dt(df: pd.DataFrame) -> pd.DataFrame:
    """将整数索引转换为日期索引"""
    if isinstance(df.index, pd.MultiIndex):
        if not isinstance(df.index.levels[0], pd.DatetimeIndex):
            df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0], format='%Y%m%d'), level=0)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, format='%Y%m%d')
    
    return df


def set_index_from_dt_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """将日期索引转换为整数索引"""
    if isinstance(df.index, pd.MultiIndex):
        if isinstance(df.index.levels[0], pd.DatetimeIndex):
            df.index = df.index.set_levels(df.index.levels[0].strftime('%Y%m%d').astype(int), level=0)
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime('%Y%m%d').astype(int)
    
    return df


# ==================== IC计算函数 ====================
def t_ic(f_, ret_mat):
    """
    计算时序IC
    """
    f_ = f_.loc[:, (f_ != 0).any(axis=0)]
    f_ = f_.dropna(how='all', axis=1)
    tic_result = pd.DataFrame(columns=['Symbol', 'tic', 'rtic', 'tic_dif', 'rtic_dif', 'tic_dif_rolling_mean', 'ticir_rolling_dif'])
    
    if not isinstance(f_.index, pd.core.indexes.datetimes.DatetimeIndex):
        f_.index = f_.index.map(lambda x: pd.to_datetime(str(int(x))))
    
    for i in f_.columns:
        try:
            test = pd.concat([f_[i], ret_mat[i]], axis=1).dropna()
            test.columns = ['f', 'o2o']
            
            test['tic_dif'] = round(test['f'].rolling(window=125, min_periods=50).corr(test['o2o']), 4)
            tic = round(corr_no_demean(test.f, test.o2o), 4)
            tic_dif = test['f'].corr(test['o2o'], method='pearson')
            
            lenRollPct = 100
            f = test['f']
            rt = test['o2o']
            testDf = pd.concat([
                pd.concat([f.head(lenRollPct-1).rank(pct=True)-0.5, 
                          f.rolling(lenRollPct).apply(lambda x: x.rank(pct=True).iat[-1]).dropna() - 0.5]), 
                rt
            ], axis=1).dropna()
            
            rtic = round(corr_no_demean(testDf.f, testDf.o2o), 4)
            rtic_dif = testDf['f'].corr(testDf['o2o'], method='spearman')
            tic_dif_mean = test['tic_dif'].mean()
            ticir_dif = test['tic_dif'].mean() / test['tic_dif'].std()
            
            tic_result.loc[len(tic_result)] = [i, tic, rtic, tic_dif, rtic_dif, tic_dif_mean, ticir_dif]
        except:
            continue
    
    return tic_result


def calc_ic(df_factor, df_ret, method='pearson'):
    """
    计算IC
    df_factor：因子矩阵(T *N)
    df_ret：价格序列(T *N)
    method：ic的计算方法：'spearman', 'pearson'
    """
    df_factor = df_factor.loc[:, (df_factor != 0).any(axis=0)]
    df_factor = df_factor.dropna(how='all', axis=1)
    
    df_factor, df_ret = align_matrices(df_factor, df_ret.dropna(how='all'))
    
    # 减均值的IC
    ic = df_factor.corrwith(df_ret, axis=1, method='pearson')
    
    # 不减均值IC
    ic_no_dif = pd.DataFrame(index=df_factor.index, columns=['ic'])
    for date in df_factor.index:
        corr = round(corr_no_demean(df_factor.loc[date].fillna(0), df_ret.loc[date].fillna(0)), 4)
        ic_no_dif.loc[date]['ic'] = corr
    
    # Rank IC
    ric = df_factor.corrwith(df_ret, axis=1, method='spearman')
    
    # 统计指标
    ic_mean, ic_std = ic.mean(), ic.std()
    icir = ic_mean / ic_std
    
    # 分年度统计
    ic_mean_ = ic.groupby(ic.index.year).mean()
    ic_std_ = ic.groupby(ic.index.year).std()
    icir_ = ic_mean_ / ic_std_
    
    icir_no_dif = ic_no_dif.mean()[0] / ic_no_dif.std()[0]
    ic_no_dif_mean = ic_no_dif.mean()[0]
    ricir = ric.mean() / ric.std()
    ric_mean = ric.mean()
    
    ic_by_year = pd.concat([ic_mean_, ic_std_, icir_], axis=1)
    ic_by_year.columns = ['ic_mean', 'ic_std', 'icir']
    
    ic_all_year = pd.DataFrame([ic_mean, ic_std, icir], columns=['All'], index=['ic_mean', 'ic_std', 'icir'])
    ic_all = pd.DataFrame([ic_mean, ic_std, icir, ic_no_dif_mean, icir_no_dif, ricir, ric_mean],
                          columns=['All'], index=['ic_mean', 'ic_std', 'icir', 'ic_no_dif_mean', 'icir_no_dif', 'ricir', 'ric_mean'])
    
    ic_by_year = pd.concat([ic_by_year, ic_all_year.T], axis=0)
    
    return ic_by_year, ic, ic_all


def align_matrices(factor_df, returns_df):
    """对齐两个DataFrame"""
    aligned_factor_df, aligned_returns_df = factor_df.align(returns_df, join='inner', axis=0)
    aligned_factor_df, aligned_returns_df = aligned_factor_df.align(aligned_returns_df, join='inner', axis=1)
    return aligned_factor_df, aligned_returns_df


# ==================== 信号生成函数 ====================
def signal_df_all(df_filtered):
    """生成信号 - 全部品种"""
    return df_filtered


def signal_df_N(df_filtered, n):
    """生成信号 - 选择前N个品种"""
    df = df_filtered.copy()
    if not isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
        df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d')
    
    df.index = df.index.strftime('%Y%m%d').astype(int)
    
    # 计算20%和80%分位数
    quantile_20 = df.quantile(0.2, axis=1)
    quantile_80 = df.quantile(0.8, axis=1)
    
    # 标记
    signal_df_percent = pd.DataFrame(0, index=df.index, columns=df.columns)
    signal_df_percent[df.lt(quantile_20, axis=0)] = -1
    signal_df_percent[df.gt(quantile_80, axis=0)] = 1
    
    return signal_df_percent


def select_factor_by_columns(df_, stocknum_list):
    """按列选择因子"""
    df = df_.copy()
    if isinstance(df.index, pd.MultiIndex):
        return df.loc[pd.IndexSlice[:, stocknum_list], :].sort_index()
    else:
        df.columns = df.columns.str.lower()
        missing_stock = set(stocknum_list) - set(df.columns)
        if not missing_stock:
            return df.loc[:, stocknum_list]
        else:
            print('Missing:', missing_stock)
            for s in missing_stock:
                df.loc[:, s] = np.nan
            return df.loc[:, stocknum_list]


# ==================== 持仓转换函数 ====================
def get_pos(signal, vola, mc_val, ticktime=-1, weight="vol", cash=1e8, cashLimit=None):
    """
    每日信号到仓位转换机制
    """
    signal = signal.dropna(how='all')
    
    if weight == "equal":
        pos = (
            signal.apply(lambda x: x * cash / x.count() / mc_val.loc[x.name], axis=1)
            .dropna(how="all")
            .round()
        )
    elif weight == "vol":
        pos = (
            signal.apply(
                lambda x: x * cash / x.count() * 0.1 / vola.loc[x.name] / mc_val.loc[x.name],
                axis=1,
            )
            .dropna(how="all")
            .round()
        )
    
    if cashLimit is not None:
        mvPtfl = (pos * mc_val).dropna(how="all").abs().sum(axis=1)
        xPos = (mvPtfl / cashLimit).map(lambda x: max(1, x))
        pos = np.floor(pos.div(xPos, axis=0))
    
    posFile = pos.stack().to_frame().reset_index()
    posFile.columns = ["Date", "contract", "volume"]
    posFile.loc[:, "TickTime"] = ticktime
    
    if ticktime == 90000000:
        x = posFile[posFile.contract.isin(["if", "ih", "ic", "im"])].index
        posFile.loc[x, "TickTime"] = 93000000
    
    mvPtfl = (pos * mc_val).dropna(how="all").abs().sum(axis=1)
    
    return posFile.fillna(0), pos.fillna(0)


def signal_to_position(signal_df, start_date, end_date, varityLst, AUM=1e8, market_data=None):
    """信号转持仓"""
    data = get_fut_daily_by(start_date - 10000, end_date, varity=varityLst, data_class="mc")
    data.Volume = data.Volume.replace(0, np.nan)
    data = data.dropna(how="all", subset=["Volume"])
    data = data.set_index(["Date", "Varity"]).sort_index()
    
    settlePrice = data.SettlePrice.unstack()
    multi = all_instruments(market="future").groupby(["Varity"]).Multi.last().loc[varityLst]
    mc_val = settlePrice * multi
    vola = cal_vol((data.SettlePrice / data.PreSettlePrice - 1).unstack(), 22)
    
    signal = signal_df.replace(0, np.nan)
    posFileWtpy, pos = get_pos(
        signal,
        vola,
        mc_val,
        ticktime=-1,
        weight="vol",
        cash=1e10,
        cashLimit=1e10*1.2,
    )
    
    return pos, posFileWtpy


# ==================== 回测函数 ====================
def get_rtn(day=1, startDate=20090101, endDate=None, sym_to_test=None):
    """获取收益率"""
    if endDate is None:
        endDate = int(datetime.date.today().strftime('%Y%m%d'))
    
    if sym_to_test is None:
        sym_to_test = ['a','ag','al','ap','au','bc','bu','c','cf','cs','cu','eb','eg','fg','fu','hc','i',
                       'j','jm','l','lu','m','ma','ni','nr','oi','p','pb','pf','pg','pp','rb',
                       'rm','ru','sa','sc','sf','sm','sn','sp','sr','ss','ta','v','y','zn','ic','im','if','ih']
    
    syms = ['a', 'ag', 'al', 'ap', 'au','ao', 'b', 'bb', 'bc', 'bu', 'c', 'cf',
            'cj', 'cs', 'cu', 'eb', 'eg','ec', 'fb', 'fg', 'fu', 'hc', 'i',
            'j', 'jd', 'jm', 'jr', 'l','lc', 'lh', 'lr', 'lu', 'm',
            'ma', 'ni', 'nr', 'oi', 'p', 'pb', 'pf', 'pg', 'pk', 'pm', 'pp','pr','px', 'rb',
            'ri', 'rm', 'rr', 'rs', 'ru', 'sa', 'sc', 'sf', 'si','sm', 'sn', 'sp', 'sr',
            'ss','sh', 't', 'ta', 'tf', 'tl', 'ts', 'ur', 'v', 'wh', 'y', 'zc',
            'zn','ps','im','if','ih','ic']
    
    mkt = get_fut_daily_by(startDate, endDate, varity=syms, data_class='mc')
    varityLst = [i.lower() for i in syms]
    multi = all_instruments(market='future').groupby(['Varity']).Multi.last().loc[varityLst]
    mkt = mkt.rename(columns={'Varity':'Symbol'}).set_index(['Date', 'Symbol']).sort_index()
    priceAdj = mkt.PriceAdj.unstack().cumsum()
    OpenAdj = mkt.Open.unstack() - priceAdj
    Open = mkt.Open.unstack()
    
    ret_mat = OpenAdj.shift(-(day+1)).sub(OpenAdj.shift(-1)).div(Open.shift(-1))
    ret_mat = set_index_from_int_to_dt(ret_mat)
    ret_mat.columns = ret_mat.columns.str.lower()
    
    return ret_mat


def stat_rt(rtPfl, rtBm=None, interest='simple', plot=True, direction=1):
    """
    基于日频收益率序列统计风险指标
    """
    rtDf = rtPfl.to_frame('strat')
    
    if interest == 'compound':
        stat = pd.DataFrame([[SR2(rtPfl), annRT2(rtPfl), annVol(rtPfl), maxDD2(rtPfl)]],
                           columns=['SR', 'annRT', 'annVol', 'maxDD'], index=['pfl'])
        gp = rtPfl.groupby(rtPfl.index // 10000)
        statByYear = pd.concat([gp.apply(SR2), gp.apply(annRT2), gp.apply(annVol), gp.apply(maxDD2)], axis=1)
        statByYear.columns = ['SR', 'annRT', 'annVol', 'maxDD']
        
        if rtBm is not None:
            rtBm = rtBm.reindex(rtPfl.index)
            statBm = pd.DataFrame([[SR2(rtBm), annRT2(rtBm), annVol(rtBm), maxDD2(rtBm)]],
                                 columns=['SR', 'annRT', 'annVol', 'maxDD'], index=['bm'])
            rtAlpha = (rtPfl - rtBm).dropna() * direction
            statAlpha = pd.DataFrame([[SR2(rtAlpha), annRT2(rtAlpha), annVol(rtAlpha), maxDD2(rtAlpha)]],
                                    columns=['SR', 'annRT', 'annVol', 'maxDD'], index=['alpha'])
            stat = pd.concat([stat, statBm, statAlpha])
            
            gp = rtBm.groupby(rtBm.index // 10000)
            statByYearBm = pd.concat([gp.apply(SR2), gp.apply(annRT2), gp.apply(annVol), gp.apply(maxDD2)], axis=1)
            statByYearBm.columns = ['SR_bm', 'annRT_bm', 'annVol_bm', 'maxDD_bm']
            
            gp = rtAlpha.groupby(rtAlpha.index // 10000)
            statByYearAlpha = pd.concat([gp.apply(SR2), gp.apply(annRT2), gp.apply(annVol), gp.apply(maxDD2)], axis=1)
            statByYearAlpha.columns = ['SR_alpha', 'annRT_alpha', 'annVol_alpha', 'maxDD_alpha']
            
            statByYear = pd.concat([statByYear, statByYearBm, statByYearAlpha], axis=1).dropna()
            rtDf = pd.concat([rtDf, rtBm, rtAlpha], axis=1)
            rtDf.columns = ['strat', 'bm', 'alpha']
        
        nvDf = (rtDf + 1).cumprod()
        
    elif interest == 'simple':
        stat = pd.DataFrame([[SR(rtPfl), annRT(rtPfl), annVol(rtPfl), maxDD(rtPfl)]],
                           columns=['SR', 'annRT', 'annVol', 'maxDD'], index=['pfl'])
        
        gp = rtPfl.groupby(rtPfl.index // 10000)
        statByYear = pd.concat([gp.apply(SR), gp.apply(annRT), gp.apply(annVol), gp.apply(maxDD)], axis=1)
        statByYear.columns = ['SR', 'annRT', 'annVol', 'maxDD']
        
        gp_month = rtPfl.groupby((rtPfl.index // 100).astype(str))
        statByMonth = pd.concat([gp_month.apply(SR), gp_month.apply(annRT), gp_month.apply(annVol), gp_month.apply(maxDD)], axis=1)
        statByMonth.columns = ['SR', 'annRT', 'annVol', 'maxDD']
        
        if rtBm is not None:
            rtBm = rtBm.reindex(rtPfl.index)
            statBm = pd.DataFrame([[SR(rtBm), annRT(rtBm), annVol(rtBm), maxDD(rtBm)]],
                                 columns=['SR', 'annRT', 'annVol', 'maxDD'], index=['bm'])
            rtAlpha = (rtPfl - rtBm).dropna() * direction
            statAlpha = pd.DataFrame([[SR(rtAlpha), annRT(rtAlpha), annVol(rtAlpha), maxDD(rtAlpha)]],
                                    columns=['SR', 'annRT', 'annVol', 'maxDD'], index=['alpha'])
            stat = pd.concat([stat, statBm, statAlpha])
            
            gp = rtBm.groupby(rtBm.index // 10000)
            statByYearBm = pd.concat([gp.apply(SR), gp.apply(annRT), gp.apply(annVol), gp.apply(maxDD)], axis=1)
            statByYearBm.columns = ['SR_bm', 'annRT_bm', 'annVol_bm', 'maxDD_bm']
            
            gp = rtAlpha.groupby(rtAlpha.index // 10000)
            statByYearAlpha = pd.concat([gp.apply(SR), gp.apply(annRT), gp.apply(annVol), gp.apply(maxDD)], axis=1)
            statByYearAlpha.columns = ['SR_alpha', 'annRT_alpha', 'annVol_alpha', 'maxDD_alpha']
            
            statByYear = pd.concat([statByYear, statByYearBm, statByYearAlpha], axis=1).dropna()
            
            gp_month = rtBm.groupby((rtBm.index // 100).astype(str))
            statByMonthBm = pd.concat([gp_month.apply(SR), gp_month.apply(annRT), gp_month.apply(annVol), gp_month.apply(maxDD)], axis=1)
            statByMonthBm.columns = ['SR_bm', 'annRT_bm', 'annVol_bm', 'maxDD_bm']
            statByMonth = pd.concat([statByMonth, statByMonthBm], axis=1).dropna()
            
            rtDf = pd.concat([rtDf, rtBm, rtAlpha], axis=1)
            rtDf.columns = ['strat', 'bm', 'alpha']
        
        nvDf = rtDf.cumsum() + 1
    else:
        print('error: interest mode is not definded!')
        return
    
    if plot:
        nvDf.index = pd.to_datetime(nvDf.index, format='%Y%m%d')
        nvDf.plot(figsize=(30, 8), grid='on')
    
    return stat, statByYear, statByMonth


def backtest_daily(pos, close, Open=None, closeBm=None, tradeTime='nextOpen', interest='simple', 
                  costRate=0.0005, direction=1, plot=True, logy=False, printSw=True, 
                  multiDict={'ih': 300, 'if': 300, 'ic': 200, 'im': 200}):
    """
    基于目标仓位计算日频收益率及各项风险指标
    """
    close = close.reindex(pos.index)
    Open = Open.reindex(pos.index) if Open is not None else None
    
    if interest == 'compound':
        if tradeTime == 'nextOpen':
            jump = Open.div(close.shift()) - 1
            intra = close.div(Open) - 1
            rt = (pos.shift(1).fillna(0) * intra + 1) * (pos.shift(2).fillna(0) * jump + 1) - 1
            posOld = pos.shift(2) * (1 + intra.shift(1))
            posOld = posOld.div(posOld.sum(axis=1), axis=0)
            posDiff = pos.shift(1) - posOld
        elif tradeTime == 'close':
            daily = close.pct_change()
            rt = pos.shift(1).fillna(0) * daily
            posOld = pos.shift(1) * (1 + close.pct_change())
            posOld = posOld.div(posOld.sum(axis=1), axis=0)
            posDiff = pos - posOld
        else:
            print('Error: tradeTime is wrong!')
            return
        
        rt_sum = rt.sum(axis=1)
        rt_sum = rt_sum - posDiff.abs().sum(axis=1).fillna(0) * costRate * direction / 2.
        rtBm = closeBm.pct_change().reindex(rt.index) if closeBm is not None else None
        avgRd_sum = posDiff.abs().sum(axis=1).mean() / 2
        
    elif interest == 'simple':
        if tradeTime == 'nextOpen':
            intra = close - Open
            jump = Open - close.shift()
            pnl = pos.shift(1).fillna(0) * intra + pos.shift(2).fillna(0) * jump
        elif tradeTime == 'close':
            daily = close.diff()
            pnl = pos.shift(1).fillna(0) * daily
        else:
            print('Error: tradeTime is wrong!')
            return
        
        multi = {x: 1 for x in pos.columns}
        multi.update({x: multiDict[x] for x in multiDict if x in multi})
        
        pnl = pnl.mul(multi)
        pnl_sum = pnl.sum(axis=1)
        tv = (pos.diff().abs() * close).mul(multi).shift().fillna(0)
        tv_sum = (pos.diff().abs() * close).mul(multi).sum(axis=1).shift().fillna(0)
        pnl = pnl - tv * costRate * direction / 2
        pnl_sum = pnl_sum - tv_sum * costRate * direction / 2.
        
        cash = pnl.std() * (250 ** 0.5) * 10
        cash_sum = pnl_sum.std() * (250 ** 0.5) * 10
        avgRd = tv.mean() / 2 / cash
        avgRd_sum = tv_sum.mean() / 2 / cash_sum
        rt = pnl / cash
        rt_sum = pnl_sum / cash_sum
        
        if printSw:
            print('cash of strat is %.1f M.' % (cash_sum / 1000000.))
        
        if closeBm is not None:
            pnlBm = closeBm.diff()
            pnlBm = pnlBm.reindex(rt_sum.index)
            rtBm = pnlBm / (pnlBm.std() * (250 ** 0.5) * 10)
        else:
            rtBm = None
    else:
        print('Error: interest is wrong!')
        return
    
    rt_sum = rt_sum.dropna()
    longRatio = pos.sum(axis=1).mean()
    
    stat, statByYear, statByMonth = stat_rt(rt_sum, rtBm=rtBm, interest=interest, plot=False, direction=direction)
    
    stat = pd.DataFrame()
    rt_ = rt_sum.copy()
    rt_std = rt_.std()
    rt_mean = rt_.mean()
    rt_cumsum = rt_.cumsum()
    rt_dd = rt_cumsum.expanding().max() - rt_cumsum
    
    stat.loc[0, 'sr'] = rt_mean / (rt_std if rt_std != 0.0 else np.nan) * (250 ** 0.5)
    stat.loc[0, 'annRt'] = rt_mean * 250
    stat.loc[0, 'maxDD'] = rt_dd.max()
    stat.loc[0, 'annVol'] = rt_std * (250 ** 0.5)
    stat.loc[0, 'cash(M)'] = round(cash_sum/1000000, 2)
    stat.loc[0, 'gain(BP)'] = pnl_sum.sum() / tv_sum.sum() * 10000
    stat.loc[0, 'calmar'] = stat.at[0, 'annRt'] / stat.at[0, 'maxDD']
    stat.loc[0, 'skew'] = skew(rt_)
    stat.loc[0, 'kurtosis'] = kurtosis(rt_)
    
    if plot:
        pos = pos.reindex(rt_sum.index)
        if interest == 'compound':
            nv = (rt_sum + 1).cumprod()
            if closeBm is not None:
                nvBm = (rtBm + 1).cumprod()
        elif interest == 'simple':
            nv = rt_sum.cumsum() + 1
            if closeBm is not None:
                nvBm = rtBm.cumsum() + 1
        
        posSum = pos.sum(axis=1)
        posCount = pos[pos != 0].count(axis=1)
        posSum.name, posCount.name = 'posSum', 'posCount'
        
        if closeBm is not None:
            rtAlphaPlot = (rt_sum - rtBm) * direction
            if interest == 'compound':
                nvAlpha = (rtAlphaPlot + 1).cumprod()
            elif interest == 'simple':
                nvAlpha = rtAlphaPlot.cumsum() + 1
            p = pd.concat([nv, nvBm, nvAlpha], axis=1)
            p.columns = ['nv', 'mkt', 'alpha']
            p.index = pd.to_datetime(p.index, format='%Y%m%d')
            p.plot(title='nav', secondary_y=['alpha'], logy=logy, figsize=(30, 8), grid='on')
        else:
            p = nv.to_frame('nv')
            p.index = pd.to_datetime(p.index, format='%Y%m%d')
            p.plot(title='nav', logy=logy, figsize=(30, 8), grid='on')
        
        p2 = pd.concat([posSum, posCount], axis=1)
        p2.index = pd.to_datetime(p2.index, format='%Y%m%d')
        p2.plot(title='pos', secondary_y=['posCount'], logy=logy, figsize=(30, 8), grid='on')
    
    return rt_sum, avgRd_sum, longRatio, stat, statByYear, statByMonth, rt


def volume_backtest(pos, costRate, tradeTime='nextOpen'):
    """执行回测"""
    pos.columns = pos.columns.str.lower()
    mkt = get_fut_daily_by(pos.index.min(), pos.index.max(), varity=pos.columns.tolist(), data_class='mc',
                          cols=['Variety','Date','Open','LastPrice','SettlePrice','PriceAdj'])
    varityLst = [i.lower() for i in pos.columns.tolist() if i in ['a', 'ag', 'al', 'ap', 'au','ao', 'b', 'bb', 'bc', 'bu', 'c', 'cf',
                                                                    'cj', 'cs', 'cu', 'eb', 'eg','ec', 'fb', 'fg', 'fu', 'hc', 'i',
                                                                    'j', 'jd', 'jm', 'jr', 'l','lc', 'lh', 'lr', 'lu', 'm',
                                                                    'ma', 'ni', 'nr', 'oi', 'p', 'pb', 'pf', 'pg', 'pk', 'pm', 'pp','pr','px', 'rb',
                                                                    'ri', 'rm', 'rr', 'rs', 'ru', 'sa', 'sc', 'sf', 'si','sm', 'sn', 'sp', 'sr',
                                                                    'ss','sh', 't', 'ta', 'tf', 'tl', 'ts', 'ur', 'v', 'wh', 'y', 'zc',
                                                                    'zn','ps','im','if','ih','ic']]
    multi = all_instruments(market='future').groupby(['Varity']).Multi.last().loc[varityLst]
    mkt = mkt.set_index(['Date', 'Variety']).sort_index()
    priceAdj = mkt.PriceAdj.unstack().cumsum()
    openAdj = mkt.Open.unstack() - priceAdj
    settlePriceAdj = mkt.SettlePrice.unstack() - priceAdj
    
    rt_sum, avgRd, longRatio, stat2, statByYear2, statByMonth2, rt = backtest_daily(
        pos, close=settlePriceAdj, Open=openAdj, closeBm=None, tradeTime=tradeTime, 
        interest='simple', costRate=costRate, plot=False, logy=False, printSw=False, multiDict=dict(multi)
    )
    
    return rt_sum, avgRd, longRatio, stat2, statByYear2, statByMonth2, rt


# ==================== 单因子回测主函数 ====================
def single_cta_factor_backtest(f_, cfg, market_data, print_=True, plot=True):
    """
    单因子IC分析
    """
    if f_.index.nlevels > 1:
        f = f_.unstack()
    else:
        f = f_.copy()
    
    result_dict = {}
    t_pipeline_start = time.time()
    
    # 准备因子
    if not isinstance(f.index, pd.core.indexes.datetimes.DatetimeIndex):
        f.index = f.index.map(lambda x: pd.to_datetime(str(int(x))))
    try:
        f.columns = f.columns.get_level_values(1)
    except:
        pass
    f.columns = f.columns.str.lower()
    
    # 去除全空行
    all_nan_rows_idx = f[f.isna().sum(1) == f.shape[1]].index
    f.drop(all_nan_rows_idx, inplace=True)
    
    assert f.shape[0] != 0, "factor matrix dont have any value"
    
    # 计算IC
    ret_mat = market_data['ret_ctc'].shift(-2)
    ret_mat.columns = ret_mat.columns.str.lower()
    ret_mat.index = ret_mat.index.to_series().apply(lambda x: pd.to_datetime(x, format="%Y%m%d").strftime("%Y-%m-%d"))
    ret_mat.index = pd.to_datetime(ret_mat.index)
    
    ic_by_year, ic, ic_all = calc_ic(f, ret_mat)
    result_dict['ic_by_year'] = ic_by_year
    result_dict['ic'] = ic
    result_dict['ic_all'] = ic_all
    
    # 计算TIC
    tic_result = t_ic(f, ret_mat)
    result_dict['ic_t'] = tic_result
    
    result_toshow = result_dict['ic_all'].T
    result_toshow[result_dict['ic_t'].set_index(['Symbol']).columns] = result_dict['ic_t'].set_index(['Symbol']).mean().T
    
    if plot == True:
        print(' 标的因子暴露图')
        fig = plt.figure(figsize=(14, 6))
        plt.bar(result_dict['ic_t'].set_index(['Symbol']).index, 
               result_dict['ic_t'].set_index(['Symbol'])['rtic'], color='blue', alpha=0.7)
        plt.title('RTIC')
        plt.xlabel('Symbol')
        plt.ylabel('Value')
        plt.grid(False)
        plt.tight_layout()
        plt.show()
        print(' 整体IC表现')
        display(result_toshow)
        print(' 分年度IC表现')
        display(result_dict['ic_by_year'])
    
    return ic_all


def single_factor_all_contract_backtest(f_, cfg, market_data, print_=True, plot=True):
    """
    单因子完整回测
    """
    start_date = cfg.start_date
    end_date = cfg.end_date
    f_name = cfg.f_name
    r_len = cfg.r_len
    AUM = cfg.AUM
    file_address = cfg.file_address
    twap_sapn = cfg.twap_sapn
    twap_part = cfg.twap_part
    right_off_flag = cfg.right_off_flag
    fee = cfg.fee
    st = time.time()
    wtpy = cfg.wtpy
    local = cfg.local
    tradeTime = cfg.tradeTime
    n = cfg.n
    factor_direction = cfg.factor_direction
    costRate = cfg.costRate
    percent = cfg.percent
    varityLst = cfg.varityLst
    
    if print_ == True:
        print(f"回测时段{cfg.start_date}->{cfg.end_date}")
    
    t_pipeline_start = time.time()
    
    # 准备因子
    if f_.index.nlevels > 1:
        f = f_.unstack()
    else:
        f = f_.copy()
    
    if not isinstance(f.index, pd.core.indexes.datetimes.DatetimeIndex):
        f.index = f.index.map(lambda x: pd.to_datetime(str(int(x))))
    try:
        f.columns = f.columns.get_level_values(1)
    except:
        pass
    f.columns = f.columns.str.lower()
    
    all_nan_rows_idx = f[f.isna().sum(1) == f.shape[1]].index
    f.drop(all_nan_rows_idx, inplace=True)
    
    assert f.shape[0] != 0, "factor matrix dont have any value"
    
    t_bt_start = time.time()
    if print_ == True:
        print(f'扣费设置为{costRate}，交易时间设置为{tradeTime}')
    
    factor_df = f.copy()
    factor_df.index = factor_df.index.strftime('%Y%m%d').astype(int)
    
    # 筛选日期
    df_filtered = select_factor_by_columns(factor_df.loc[start_date:end_date], varityLst)
    
    res0 = {}
    res = {}
    
    signal_df_all_ = df_filtered
    signal_df_percent_all = signal_df_all_.div(signal_df_all_.abs().sum(axis=1), axis=0)
    turnoverbydate = (signal_df_percent_all - signal_df_percent_all.shift(1)).abs().sum(axis=1)
    turnoverbysym = (signal_df_percent_all - signal_df_percent_all.shift(1)).abs().sum()
    pos, pos_wtpy = signal_to_position(signal_df_all_, start_date, end_date, varityLst, AUM, market_data)
    pos = pos.replace([np.inf, -np.inf, np.nan], 0, inplace=False)
    
    res0['rt'], res0['avgRD'], res0['longRatio'], res0['stat2'], res0['statByYear2'], res0['statByMonth2'], res0['rt_sym'] = volume_backtest(pos, 0, tradeTime)
    rtn0 = res0['rt'].copy()
    rtn0.index = pd.to_datetime(rtn0.index, format='%Y%m%d')
    
    res['rt'], res['avgRD'], res['longRatio'], res['stat2'], res['statByYear2'], res['statByMonth2'], res['rt_sym'] = volume_backtest(pos, costRate, tradeTime)
    rtn = res['rt'].copy()
    rtn.index = pd.to_datetime(rtn.index, format='%Y%m%d')
    
    if plot == True:
        plt.figure(1)
        (rtn0 + 1).cumprod().plot(figsize=(30, 12), grid='on', label='NoFee')
        (rtn + 1).cumprod().plot(figsize=(30, 12), grid='on', label=f'Fee{costRate}')
        plt.title('Return cumsum all')
        plt.xlabel('Date')
        plt.ylabel('ReturnCumsum')
        plt.legend()
        plt.show()
        
        # rtn0_sym = res0['rt_sym'].copy()
        # rtn0_sym.index = pd.to_datetime(rtn0_sym.index, format='%Y%m%d')
        # sr_by_sym = {}
        # 
        # for sym in rtn0_sym.columns:
        #     (rtn0_sym[sym] + 1).cumprod().plot(figsize=(30, 8), grid='on')
        
        # plt.title('Cumulative Returns for Each Symbol')
        # plt.xlabel('Date')
        # plt.ylabel('Cumulative Returns')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
    
    res0_display = res0['stat2'].copy().rename(index={0: 'NoFee'})
    res_display = res['stat2'].copy().rename(index={0: f'Fee{costRate}'})
    if print_ == True:
        print(' 不设置扣费回测表现')
        display(res0_display)
        print(f' 设置扣费为{costRate}回测表现')
        display(res_display)
    
    return res0['stat2'], res['stat2'], turnoverbydate.mean()


# ==================== 滚动窗口主函数 ====================
def main_window_optimization(factor, windows, cfg, market_data, print_=True, plot=False):
    """
    滚动窗口优化主函数
    
    Args:
        factor: 原始因子DataFrame
        windows: 窗口列表
        cfg: 配置对象
        market_data: 市场数据字典
        print_: 是否打印
        plot: 是否绘图
    
    Returns:
        汇总结果DataFrame
    """
    factor = factor.copy().fillna(0)
    factor_standard = rolling_vol_settle(factor)
    
    ics = pd.DataFrame()
    sr0s = pd.DataFrame()
    srs = pd.DataFrame()
    
    for i in windows:
        if print_:
            print(f'正在测试窗口={i}')
        
        # 应用滚动窗口
        factor_i = factor_standard.rolling(i).mean()
        
        # IC分析
        ic = single_cta_factor_backtest(factor_i, cfg, market_data, print_=False, plot=False).T
        ic['windows'] = i
        cols = ['windows'] + [col for col in ic if col != 'windows']
        ic = ic[cols]
        
        # 回测
        sr0, sr, turnover = single_factor_all_contract_backtest(factor_i, cfg, market_data, print_=False, plot=False)
        sr0['windows'] = i
        cols = ['windows'] + [col for col in sr0 if col != 'windows']
        sr0 = sr0[cols]
        
        sr['windows'] = i
        cols = ['windows'] + [col for col in sr if col != 'windows']
        sr = sr[cols]
        sr['turnover'] = turnover
        
        ics = pd.concat([ics, ic])
        sr0s = pd.concat([sr0s, sr0])
        srs = pd.concat([srs, sr])
    
    ic_result = ics.iloc[:, 0:4].set_index('windows')
    sr0_result = sr0s.iloc[:, [0, 1, 2, 3, 7]].set_index('windows')
    sr_result = srs.iloc[:, [0, 1, 2, 3, 7, 10]].set_index('windows')
    sr_result.columns = ['sr_0.5‰', 'annRt_0.5‰', 'maxDD_0.5‰', 'calmar_0.5‰', 'turnover']
    
    result0 = pd.concat([ic_result, sr0_result], axis=1)
    result = pd.concat([result0, sr_result], axis=1)
    
    if print_:
        print("\n" + "="*60)
        print("滚动窗口优化结果汇总")
        print("="*60)
        display(result)
    
    return result


# ==================== 统一接口类 ====================
class CTABacktestFramework:
    """
    CTA回测框架统一接口
    """
    
    def __init__(self, config=None):
        """
        初始化回测框架
        
        Args:
            config: BacktestConfig配置对象
        """
        self.cfg = config if config else BacktestConfig()
        
        # 加载市场数据
        print("初始化回测框架，加载市场数据...")
        self.market_data = load_market_data(
            self.cfg.start_date,
            self.cfg.end_date,
            self.cfg.syms
        )
        print("框架初始化完成\n")
    
    def run(self, factor, use_window=None, windows=None, print_result=True, plot=True):
        """
        运行回测
        
        Args:
            factor: 因子DataFrame
            use_window: 是否使用滚动窗口优化，默认使用配置中的设置
            windows: 窗口列表，默认使用配置中的设置
            print_result: 是否打印结果
            plot: 是否绘图
        
        Returns:
            结果字典
        """
        # 确定是否使用窗口
        if use_window is None:
            use_window = self.cfg.use_window
        
        if windows is None:
            windows = self.cfg.windows
        
        if use_window:
            # 滚动窗口优化模式
            print("="*60)
            print("模式: 滚动窗口优化")
            print(f"窗口列表: {windows}")
            print("="*60 + "\n")
            
            result = main_window_optimization(
                factor,
                windows,
                self.cfg,
                self.market_data,
                print_=print_result,
                plot=plot
            )
            
            return {
                'mode': 'window_optimization',
                'result': result
            }
        else:
            # 单次回测模式
            print("="*60)
            print("模式: 单次回测")
            print("="*60 + "\n")
            
            # 标准化因子
            factor_std = rolling_vol_settle(factor.fillna(0))
            
            # IC分析
            print("\n[1/2] IC分析...")
            ic_result = single_cta_factor_backtest(
                factor_std,
                self.cfg,
                self.market_data,
                print_=print_result,
                plot=plot
            )
            
            # 回测
            print("\n[2/2] 回测执行...")
            bt_result = single_factor_all_contract_backtest(
                factor_std,
                self.cfg,
                self.market_data,
                print_=print_result,
                plot=plot
            )
            
            return {
                'mode': 'single_backtest',
                'ic_analysis': ic_result,
                'backtest_no_cost': bt_result[0],
                'backtest_with_cost': bt_result[1],
                'turnover': bt_result[2]
            }
    
    def get_market_data(self, data_type='CloseAdj'):
        """
        获取市场数据
        
        Args:
            data_type: 数据类型，可选: 'CloseAdj', 'OpenAdj', 'ret_ctc'等
        
        Returns:
            DataFrame
        """
        return self.market_data.get(data_type)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    """
    使用示例
    """
    print("CTA回测框架加载完成")
    print("\n使用方法:")
    print("1. 创建配置: config = BacktestConfig()")
    print("2. 初始化框架: framework = CTABacktestFramework(config)")
    print("3. 准备因子: factor = framework.market_data['CloseAdj'].pct_change(20)")
    print("4. 运行回测: result = framework.run(factor, use_window=False)")
    print("5. 或窗口优化: result = framework.run(factor, use_window=True, windows=[5,10,20])")