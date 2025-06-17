import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple

def align_time_series_fast(df, fill_method='ffill', fill_value=0):
    """
    Fast align time series data for different symbols without lookahead.
    Uses pivot/groupby method for better performance.
    
    Parameters:
    - df: DataFrame with 'open_time' and 'symbol' columns
    - fill_method: Method to fill missing values ('ffill' for forward fill)
    - fill_value: Value to use when there are no previous values (default: 0)
    
    Returns:
    - DataFrame with aligned time series
    """
    # Make sure open_time is datetime
    df['open_time'] = pd.to_datetime(df['open_time'])
    
    # Create a multi-index from all combinations of time and symbol
    all_times = df['open_time'].unique()
    all_symbols = df['symbol'].unique()
    
    # Create a complete index
    idx = pd.MultiIndex.from_product([all_times, all_symbols], names=['open_time', 'symbol'])
    
    # Set multi-index and reindex to get all combinations
    df_indexed = df.set_index(['open_time', 'symbol'])
    aligned_df = df_indexed.reindex(idx)
    
    # Group by symbol and apply forward fill
    if fill_method == 'ffill':
        aligned_df = aligned_df.groupby(level='symbol').ffill()
    
    # Fill remaining NaN with fill_value
    aligned_df = aligned_df.fillna(fill_value)
    
    # Reset index
    aligned_df = aligned_df.reset_index()
    
    return aligned_df.sort_values(['symbol', 'open_time']).reset_index(drop=True)


#efficient version using vectorized operations across all columns at once
"""
def minmaxscaler_by_symbol(df, feature_range=(-1, 1), target_columns=None, group_by_column='symbol'):
    
    Ultra-efficient MinMaxScaler using vectorized groupby operations
    
    Parameters:
    - df: DataFrame to scale
    - feature_range: tuple of (min, max) to scale data to, default (-1, 1)
    - target_columns: list of specific columns to scale
    - group_by_column: column to group by for scaling
    
    Returns:
    - Scaled DataFrame
    # Determine columns to scale
    if target_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        target_columns = [col for col in numeric_columns if col not in ['open_time', group_by_column]]
    
    # Create a copy
    scaled_df = df.copy()
    
    # Get feature range parameters
    feature_min, feature_max = feature_range
    feature_span = feature_max - feature_min
    
    # Get the subset of columns to scale
    data_to_scale = df[target_columns]
    
    # Perform vectorized groupby operations
    grouped = df.groupby(group_by_column)
    
    # Calculate min and max for all columns at once
    group_mins = grouped[target_columns].transform('min')
    group_maxs = grouped[target_columns].transform('max')
    group_ranges = group_maxs - group_mins
    
    # Apply scaling formula vectorized across all columns
    # Handle division by zero for constant values
    scaled_values = np.where(
        group_ranges == 0,
        feature_min,
        (data_to_scale - group_mins) / group_ranges * feature_span + feature_min
    )
    
    # Assign scaled values back to the DataFrame
    scaled_df[target_columns] = scaled_values
    
    return scaled_df
"""



class GroupMinMaxScaler:
    """
    Ultra-efficient MinMaxScaler using vectorized groupby operations
    
    Parameters:
    - df: DataFrame to scale
    - feature_range: tuple of (min, max) to scale data to, default (-1, 1)
    - target_columns: list of specific columns to scale
    - group_by_column: column to group by for scaling
    """
    def __init__(self, feature_range=(-1, 1), target_columns=None, group_by_column='symbol'):
        self.feature_min, self.feature_max = feature_range
        self.feature_span = self.feature_max - self.feature_min
        self.target_columns = target_columns
        self.group_by = group_by_column
        # 下面两个在 fit() 时会被赋值
        self._mins = None    # DataFrame: index=symbol, cols=target_columns
        self._ranges = None  # DataFrame: index=symbol, cols=target_columns

    def fit(self, df):
        # 自动选列
        if self.target_columns is None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            self.target_columns = [c for c in numeric if c not in [self.group_by]]
        # 计算每组的 min/max/range
        grp = df.groupby(self.group_by)[self.target_columns]
        self._mins   = grp.min()
        self._maxs   = grp.max()
        self._ranges = self._maxs - self._mins
        return self

    def transform(self, df):
        # 先把对应 symbol 的 mins 和 ranges merge 进来
        # Merge mins
        mins = (
            df[[self.group_by]]
            .merge(self._mins, left_on=self.group_by, right_index=True, how='left')
            .set_index(df.index).fillna(self.feature_min)
        )
        ranges = (
            df[[self.group_by]]
            .merge(self._ranges, left_on=self.group_by, right_index=True, how='left')
            .set_index(df.index).fillna(self.feature_span)
        )

        scaled = df.copy()
        data = df[self.target_columns]

        # 公式：(x - min) / range * span + feature_min, 对 range=0 的列直接赋 value=feature_min
        scaled_vals = (data - mins[self.target_columns]) \
                      .divide(ranges[self.target_columns].replace(0, np.nan)) \
                      .multiply(self.feature_span) \
                      .add(self.feature_min) \
                      .fillna(self.feature_min)

        scaled[self.target_columns] = scaled_vals
        return scaled

    def fit_transform(self, df):
        return self.fit(df).transform(df)




def pivot_features_and_costs(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    """
    Pivot DataFrame into feature tensor and cost matrix.

    Returns
    -------
    features : np.ndarray, shape (T, N, k)
    costs    : np.ndarray, shape (T, N)
    times    : list of T timestamps
    symbols  : list of N symbols
    """
    # 检查一下重复值
    duplicates = df.groupby(['open_time', 'symbol']).size()
    if (duplicates > 1).any():
        print("Warning: Found duplicate (time, symbol) pairs")
        print("Duplicates sample:")
        print(duplicates[duplicates > 1].head())
    
    # Pivot cost (target) matrix
    cost_pivot = df.pivot_table(
        values=y_col,
        index="open_time",
        columns="symbol",
        aggfunc="first"
    )
    # Sort and fill missing
    cost_pivot = cost_pivot.sort_index().fillna(0)
    unique_times = cost_pivot.index.tolist()
    unique_symbols = cost_pivot.columns.tolist()
    costs = cost_pivot.values

    # Pivot features and stack
    mats = []
    for x in x_cols:
        mat = df.pivot_table(
            values=x,
            index="open_time",
            columns="symbol",
            aggfunc="first"
        )
        mat = mat.sort_index().reindex(columns=unique_symbols).fillna(0)
        mats.append(mat.values)

    features = np.stack(mats, axis=2)
    print(f"Data shape: features {features.shape}, costs {costs.shape}")

    return features, costs, unique_times, unique_symbols


def truncate_outliers(df, truncation_thres=0.05, group_by_column='symbol', exclude_columns=None):
    """
    对每个symbol的每一列进行outlier truncation
    
    参数
    -----
    df : pandas.DataFrame
        包含多个symbol数据的DataFrame
    truncation_thres : float, optional
        截断阈值，默认0.05（即5%和95%分位数）
    group_by_column : str, optional
        分组列名，默认'symbol'
    exclude_columns : list, optional
        不进行truncation的列名列表
        
    返回
    -----
    pandas.DataFrame
        截断后的DataFrame
    """
    if exclude_columns is None:
        exclude_columns = ['symbol', 'open_time']
    
    df_truncated = df.copy()
    
    # 获取所有需要处理的数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    columns_to_process = [col for col in numeric_columns if col not in exclude_columns]
    
    print(f"对 {len(columns_to_process)} 列进行outlier truncation...")
    
    for col in columns_to_process:
        if col in df.columns:
            # 对每个symbol分别计算分位数并截断
            for symbol in df[group_by_column].unique():
                symbol_mask = df[group_by_column] == symbol
                symbol_data = df.loc[symbol_mask, col]
                
                # 计算分位数
                lower_quantile = symbol_data.quantile(truncation_thres)
                upper_quantile = symbol_data.quantile(1 - truncation_thres)
                
                # 进行截断
                df_truncated.loc[symbol_mask, col] = symbol_data.clip(
                    lower=lower_quantile, 
                    upper=upper_quantile
                )
    
    print(f"Outlier truncation完成，处理了 {len(columns_to_process)} 列")
    return df_truncated


def compute_technical_features(df, lookback_window=None, group_by_column='symbol', 
                             price_col='close', volume_col='volume', time_col='open_time'):
    """
    计算技术指标特征
    
    参数
    -----
    df : pandas.DataFrame
        包含价格和成交量数据的DataFrame
    lookback_window : int, optional
        回望窗口长度，如果为None则从config导入LOOKBACK
    group_by_column : str, optional
        分组列名，默认'symbol'
    price_col : str, optional
        价格列名，默认'close'
    volume_col : str, optional
        成交量列名，默认'volume'
    time_col : str, optional
        时间列名，默认'open_time'
        
    返回
    -----
    pandas.DataFrame
        添加了技术指标特征的DataFrame
    """
    if lookback_window is None:
        from config import LOOKBACK
        lookback_window = LOOKBACK
    
    df_features = df.copy()
    
    # 确保按时间排序
    df_features = df_features.sort_values([group_by_column, time_col]).reset_index(drop=True)
    
    print(f"计算技术指标特征，lookback_window={lookback_window}...")
    
    # 为每个symbol计算技术指标
    feature_columns = []
    
    for symbol in df_features[group_by_column].unique():
        symbol_mask = df_features[group_by_column] == symbol
        symbol_data = df_features[symbol_mask].copy()
        
        if len(symbol_data) < lookback_window:
            print(f"警告: {symbol} 数据长度 {len(symbol_data)} 小于lookback_window {lookback_window}")
            continue
            
        close_prices = symbol_data[price_col]
        volumes = symbol_data[volume_col]
        
        # 计算收益率
        returns = close_prices.pct_change()
        
        # 计算滚动特征
        symbol_features = pd.DataFrame(index=symbol_data.index)
        
        # 1. 基础统计特征
        symbol_features['mean_return'] = returns.rolling(window=lookback_window, min_periods=1).mean()
        symbol_features['volatility'] = returns.rolling(window=lookback_window, min_periods=1).std()
        
        # 2. 趋势特征
        symbol_features['trend'] = (close_prices / close_prices.shift(lookback_window) - 1).fillna(0)
        
        # 3. 成交量趋势
        vol_ma_short = volumes.rolling(window=5, min_periods=1).mean()
        vol_ma_long = volumes.rolling(window=max(5, lookback_window//2), min_periods=1).mean()
        symbol_features['volume_trend'] = (vol_ma_short / vol_ma_long - 1).fillna(0)
        
        # 4. 动量指标
        symbol_features['momentum_5d'] = (close_prices / close_prices.shift(5) - 1).fillna(0)
        symbol_features['momentum_10d'] = (close_prices / close_prices.shift(10) - 1).fillna(0)
        
        # 5. RSI类指标
        gains = returns.where(returns > 0, 0).rolling(window=lookback_window, min_periods=1).sum()
        losses = (-returns.where(returns < 0, 0)).rolling(window=lookback_window, min_periods=1).sum()
        symbol_features['rsi'] = gains / (gains + losses + 1e-8)  # 避免除零
        
        # 6. 移动平均线相关特征
        ma_5 = close_prices.rolling(window=5, min_periods=1).mean()
        ma_10 = close_prices.rolling(window=10, min_periods=1).mean()
        symbol_features['price_to_ma5'] = (close_prices / ma_5 - 1).fillna(0)
        symbol_features['price_to_ma10'] = (close_prices / ma_10 - 1).fillna(0)
        
        # 7. EMA和MACD指标
        ema_12 = close_prices.rolling(window=12, min_periods=1).mean()
        ema_26 = close_prices.rolling(window=26, min_periods=1).mean()
        ema_30 = close_prices.rolling(window=30, min_periods=1).mean()
        ema_60 = close_prices.rolling(window=60, min_periods=1).mean()
        ema_120 = close_prices.rolling(window=120, min_periods=1).mean()
        
        symbol_features['close_ema_30m'] = ema_30
        symbol_features['close_ema_60m'] = ema_60
        symbol_features['macd_30_60'] = ema_30 - ema_60
        symbol_features['macd_12_26'] = ema_12 - ema_26
        symbol_features['macd_30_120'] = ema_30 - ema_120
        
        # 8. 标准化价格特征
        symbol_features['normalized_price'] = close_prices / 100.0
        
        # 将特征添加到原始数据中
        for col in symbol_features.columns:
            if col not in feature_columns:
                feature_columns.append(col)
            df_features.loc[symbol_mask, col] = symbol_features[col].values
    
    # 填充NaN值
    for col in feature_columns:
        df_features[col] = df_features[col].fillna(0)
    
    print(f"技术指标特征计算完成，新增 {len(feature_columns)} 个特征:")
    print(f"特征列表: {feature_columns}")
    
    return df_features


def truncate_target_variable(df, target_col, truncation_thres=0.05, group_by_column='symbol'):
    """
    专门对目标变量进行outlier truncation
    
    参数
    -----
    df : pandas.DataFrame
        包含目标变量的DataFrame
    target_col : str
        目标变量列名
    truncation_thres : float, optional
        截断阈值，默认0.05（即5%和95%分位数）
    group_by_column : str, optional
        分组列名，默认'symbol'
        
    返回
    -----
    pandas.DataFrame
        截断后的DataFrame
    """
    df_truncated = df.copy()
    
    if target_col not in df.columns:
        print(f"⚠️ 警告: {target_col} 列不存在于数据中")
        return df_truncated
    
    print(f"对目标变量 {target_col} 进行outlier truncation...")
    
    # 统计截断前的信息
    original_stats = df[target_col].describe()
    print(f"截断前 {target_col} 统计信息:")
    print(f"  均值: {original_stats['mean']:.6f}")
    print(f"  标准差: {original_stats['std']:.6f}")
    print(f"  最小值: {original_stats['min']:.6f}")
    print(f"  最大值: {original_stats['max']:.6f}")
    
    # 对每个symbol分别截断
    for symbol in df[group_by_column].unique():
        symbol_mask = df[group_by_column] == symbol
        symbol_data = df.loc[symbol_mask, target_col]
        
        # 计算分位数
        lower_quantile = symbol_data.quantile(truncation_thres)
        upper_quantile = symbol_data.quantile(1 - truncation_thres)
        
        # 统计会被截断的数据点数量
        n_truncated_lower = (symbol_data < lower_quantile).sum()
        n_truncated_upper = (symbol_data > upper_quantile).sum()
        total_points = len(symbol_data)
        
        if n_truncated_lower + n_truncated_upper > 0:
            print(f"  {symbol}: 截断 {n_truncated_lower + n_truncated_upper}/{total_points} 个点 "
                  f"(下限:{n_truncated_lower}, 上限:{n_truncated_upper})")
        
        # 进行截断
        df_truncated.loc[symbol_mask, target_col] = symbol_data.clip(
            lower=lower_quantile, 
            upper=upper_quantile
        )
    
    # 统计截断后的信息
    truncated_stats = df_truncated[target_col].describe()
    print(f"截断后 {target_col} 统计信息:")
    print(f"  均值: {truncated_stats['mean']:.6f}")
    print(f"  标准差: {truncated_stats['std']:.6f}")
    print(f"  最小值: {truncated_stats['min']:.6f}")
    print(f"  最大值: {truncated_stats['max']:.6f}")
    
    return df_truncated


def split_train_test_by_time_quantile(df, time_col='open_time', frac=0.8):
    """
    按 time_col 的 frac 分位点来划分训练/测试集。

    参数
    -----
    df       : pandas.DataFrame
        包含时间列（或索引）的 DataFrame。
    time_col : str, optional
        时间列名，如果 df 已经用时间索引则设为 None。
    frac     : float, optional
        用于训练集的比例，默认 0.8（即前 80% 的时间）。

    返回
    -----
    train_df, test_df : pandas.DataFrame
    """
    # 如果 open_time 是列，就用那列；否则假设 df.index 是 DatetimeIndex
    if time_col is not None:
        times = pd.to_datetime(df[time_col])
    else:
        times = pd.to_datetime(df.index)

    # 计算 80% 分位的时间点
    cutoff = times.quantile(frac)

    # 划分
    if time_col is not None:
        train_df = df[times <= cutoff]
        test_df  = df[times  > cutoff]
    else:
        train_df = df[df.index <= cutoff]
        test_df  = df[df.index  > cutoff]

    return train_df, test_df


def print_memory(stage):
    """打印内存使用"""
    import psutil
    process = psutil.Process()
    memory_gb = process.memory_info().rss / (1024**3)
    print(f" {stage}: {memory_gb:.2f} GB")
    return memory_gb


def validate_optimization_data(features, costs, N, cov_matrix, risk_f):
    """Validate data before optimization"""
    print("\n=== 数据验证 ===")
    
    # 1. Check for NaN/Inf values
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        print("❌ Features contain NaN/Inf values")
        return False
    
    if np.any(np.isnan(costs)) or np.any(np.isinf(costs)):
        print("❌ Costs contain NaN/Inf values")
        return False
    
    # 2. Check data ranges
    print(f"Features range: [{np.min(features):.4f}, {np.max(features):.4f}]")
    print(f"Costs range: [{np.min(costs):.4f}, {np.max(costs):.4f}]")
    
    # 3. Check covariance matrix
    eigenvals = np.linalg.eigvals(cov_matrix)
    min_eigenval = np.min(eigenvals)
    print(f"Covariance matrix min eigenvalue: {min_eigenval:.6f}")
    
    if min_eigenval < -1e-8:
        print("❌ Covariance matrix is not positive semi-definite")
        return False
    
    # 4. Check condition number
    cond_num = np.linalg.cond(cov_matrix)
    print(f"Covariance matrix condition number: {cond_num:.2e}")
    
    if cond_num > 1e12:
        print("⚠️  Covariance matrix is ill-conditioned")
    
    # 5. Check constraint compatibility
    from config import RISK_ABS, SINGLE_ABS, L1_ABS
    
    max_possible_l1 = N * SINGLE_ABS
    print(f"Max possible L1 norm: {max_possible_l1:.2f}, Required L1: {L1_ABS:.2f}")
    
    print("✅ 数据验证通过")
    return True


def save_model_parameters(N, costs, save_regular=True, save_testing=True):
    """
    Save market neutral model parameters to pickle files
    
    Parameters
    ----------
    N : int
        Number of assets
    costs : np.ndarray
        Cost matrix for computing covariance and risk factors
    save_regular : bool
        Whether to save regular model parameters
    save_testing : bool  
        Whether to save testing model parameters (with turnover)
    """
    from config import (get_constraint_matrices, get_covariance_matrix, get_risk_factor,
                       RISK_ABS, SINGLE_ABS, L1_ABS, SIGMA_ABS, TURNOVER,
                       MARKET_MODEL_DIR, MARKET_MODEL_DIR_TESTING)
    import pickle
    
    # Get dynamic parameters
    A, b, l, u = get_constraint_matrices(N)
    cov_matrix = get_covariance_matrix(costs)
    risk_f = get_risk_factor(costs)
    
    if save_regular:
        print("保存常规模型参数...")
        params = dict(
            N=N, A=A, b=b, l=l, u=u,
            risk_f=risk_f, risk_abs=RISK_ABS,
            single_abs=SINGLE_ABS, l1_abs=L1_ABS,
            cov_matrix=cov_matrix, sigma_abs=SIGMA_ABS
        )
        with open(MARKET_MODEL_DIR, "wb") as f:
            pickle.dump(params, f)
    
    if save_testing:
        print("保存测试模型参数...")
        params_testing = dict(
            N=N, A=A, b=b, l=l, u=u,
            risk_f=risk_f, risk_abs=RISK_ABS,
            single_abs=SINGLE_ABS, l1_abs=L1_ABS,
            cov_matrix=cov_matrix, sigma_abs=SIGMA_ABS, 
            turnover=TURNOVER
        )
        with open(MARKET_MODEL_DIR_TESTING, "wb") as f:
            pickle.dump(params_testing, f)
    
    return A, b, l, u, risk_f, cov_matrix


def process_and_combine_shared(features, costs, batch_size=1000, **override_params):
    """
    Process and combine shared memory optimization results
    """
    from multiprocessing import get_context, shared_memory
    from config import (get_constraint_matrices, get_covariance_matrix, get_risk_factor,
                       RISK_ABS, SINGLE_ABS, L1_ABS, SIGMA_ABS, LOOKBACK, PADDING_METHOD)
    from sklearn.decomposition import PCA
    from batch_runner import run_batch_shared
    import gc
    
    ctx = get_context('spawn')
    total_samples = len(features)
    all_feats, all_sols, all_objs = [], [], []

    # Get parameters from config or overrides
    N = override_params.get('N', features.shape[1])
    A, b, l, u = get_constraint_matrices(N)
    cov_matrix = get_covariance_matrix(costs)
    risk_f = get_risk_factor(costs)
    
    # Use config values unless overridden
    risk_abs = override_params.get('risk_abs', RISK_ABS)
    single_abs = override_params.get('single_abs', SINGLE_ABS)
    l1_abs = override_params.get('l1_abs', L1_ABS)
    sigma_abs = override_params.get('sigma_abs', SIGMA_ABS)
    
    if not validate_optimization_data(features, costs, N, cov_matrix, risk_f):
        raise ValueError("数据验证失败，无法继续优化")
        
    for i in range(0, total_samples, batch_size):
        start, end = i, min(i + batch_size, total_samples)
        print(f"\n 共享内存子进程处理样本 {start} 到 {end - 1}...")

        feats_batch = features[start:end]
        costs_batch = costs[start:end]

        shapes = {
            'feats': (feats_batch.shape[0], feats_batch.shape[1], LOOKBACK, feats_batch.shape[2]),
            'sols': (feats_batch.shape[0], feats_batch.shape[1]),
            'objs': (feats_batch.shape[0], 1)
        }
        dtypes = {'feats': feats_batch.dtype, 'sols': np.float32, 'objs': np.float32}

        shms = {key: shared_memory.SharedMemory(create=True, size=np.zeros(shapes[key], dtype=dtypes[key]).nbytes)
                for key in shapes}

        shm_names = {k: v.name for k, v in shms.items()}

        p = ctx.Process(
            target=run_batch_shared,
            args=(shm_names, shapes, dtypes, feats_batch, costs_batch, N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs)
        )
        p.start()
        p.join()

        feats_np = np.ndarray(shapes['feats'], dtype=dtypes['feats'], buffer=shms['feats'].buf).copy()
        sols_np = np.ndarray(shapes['sols'], dtype=dtypes['sols'], buffer=shms['sols'].buf).copy()
        objs_np = np.ndarray(shapes['objs'], dtype=dtypes['objs'], buffer=shms['objs'].buf).copy()

        all_feats.append(feats_np)
        all_sols.append(sols_np)
        all_objs.append(objs_np)

        for shm in shms.values():
            shm.close()
            shm.unlink()

        del feats_batch, costs_batch
        gc.collect()

    print("\n 合并所有批次...")
    return {
        'feats': np.concatenate(all_feats, axis=0),
        'costs': costs,
        'sols': np.concatenate(all_sols, axis=0),
        'objs': np.concatenate(all_objs, axis=0),
        'lookback': LOOKBACK,
        'padding_method': PADDING_METHOD
    }
