import torch
import numpy as np
import os


#############################################################################
# COST & FEATURE COLUMNS
#############################################################################
Y_COL = 'return_30min'  

# Original features + Technical indicators
X_COLS = [
    # Original features
    'open', 'high', 'low', 'close', 'volume', 'count',
    
    # Technical indicators from compute_technical_features
    'mean_return', 'volatility', 'trend', 'volume_trend',
    'momentum_5d', 'momentum_10d', 'rsi', 'price_to_ma5', 'price_to_ma10',
    'close_ema_30m', 'close_ema_60m', 'macd_30_60', 'macd_12_26', 'macd_30_120',
    'normalized_price'
]


#############################################################################
# FILE PATHS
#############################################################################
ROOT_PATH = "/scratch/gpfs/sl3965/datasets"
#ROOT_PATH = "/home/yuheng" # "/Users/tarothousand/Desktop/EndToEnd/my-PyEPO/00_portfolio_real_data"
RAW_DATA_PATH = os.path.join(ROOT_PATH, "perp_futures_klines")
PROCESSED_DATA_PATH = os.path.join(ROOT_PATH, "processed_crypto_data.csv")
ALIGNED_CRYPTO_DATA_PATH = os.path.join(ROOT_PATH, "aligned_crypto_data.parquet")

TRAIN_OPTDATA_DIR = "./train_data"
TEST_DATA_DIR = "./test_data"
OPTDATA_NAME = "crypto_data"
DATASET_DICT_PATH = os.path.join(ROOT_PATH, "train_market_neutral_dataset.npz")
TEST_DATASET_DICT_PATH = os.path.join(ROOT_PATH, "test_market_neutral_dataset.npz")

#############################################################################
# OPT_DATASET PRECOMPUTATION
#############################################################################
LOOKBACK = 5
PRECOMPUTE_BATCH_SIZE = 500
PADDING_METHOD = "zero"
MARKET_MODEL_DIR = "market_neutral_model_params.pkl"
MARKET_MODEL_DIR_TESTING = "market_neutral_model_params_testing.pkl"


#############################################################################
# MARKET NEUTRAL MODEL PARAMETERS
#############################################################################
# Number of assets (will be updated dynamically based on data)
N = 13

# Constraint matrices (will be updated dynamically)
def get_constraint_matrices(n_assets):
    """Generate constraint matrices based on number of assets"""
    A = np.ones((1, n_assets))
    b = np.array([0.0])
    l = np.zeros(n_assets)
    u = np.zeros(n_assets) + 1e6
    return A, b, l, u

def get_covariance_matrix(costs_data):
    """Generate covariance matrix from costs data"""
    return np.cov(costs_data, rowvar=False, bias=False)

def get_risk_factor(costs_data):
    """Generate risk factor from PCA of costs data"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    return pca.fit(costs_data).components_[0]

# Risk and constraint parameters
RISK_ABS = 1.5
SINGLE_ABS = 0.1
L1_ABS = 1.0
SIGMA_ABS = 2.5
TURNOVER = 0.5

# Data processing parameters
TRUNCATION_THRESHOLD = 0.05
TRAIN_TEST_SPLIT_RATIO = 0.8


#############################################################################
# DEVICE
#############################################################################
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
# Print extra info
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
elif DEVICE.type == "mps":
    print("Using Apple Silicon GPU via Metal Performance Shaders (MPS)")
    
    
#############################################################################
# NEURAL NETWORK PARAMETERS
#############################################################################
K = 21
HIDDEN_DIM = 32
LSTM_HIDDEN_DIM = 64
DROPOUT_RATE = 0.0

NUM_EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-3

LSTM_SAVE_DIR = "./lstm"