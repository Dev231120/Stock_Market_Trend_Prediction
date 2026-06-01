Stock Market Trend Prediction using LSTM & Sentiment Analysis-

A deep learning project that combines LSTM neural networks with NLP-based news sentiment analysis to predict Apple (AAPL) stock prices and classify next-day price direction — complete with uncertainty estimation and a backtested trading strategy.

Project Overview-
This project builds two models-

Regression Model — Predicts the next day's closing price of AAPL stock
Classification Model — Predicts whether the price will go UP (1) or DOWN (0) the next day

Both models are trained on a rich feature set combining technical indicators (EMA, RSI) with daily sentiment scores derived from real financial news headlines (CNBC, Reuters, The Guardian).

Project Structure-
stock_market_trend

├── stock_market_trend.ipynb      # Main Jupyter notebook
├── archive (3).zip               # News headlines dataset (CNBC, Reuters, Guardian)
│   ├── cnbc_headlines.csv
│   ├── guardian_headlines.csv
│   └── reuters_headlines.csv
└── README.md

Dataset-
Stock Price Data

Source: Yahoo Finance via yfinance
Ticker: AAPL (Apple Inc.)
Period: January 1, 2020 – January 1, 2025
Features used: Close, Volume

News Sentiment Data-

Sources: CNBC, Reuters, The Guardian (CSV files)
Columns standardized to: date, text
Sentiment scoring: NLTK VADER (compound score, range: -1 to +1)
All three sources are merged into a single daily_sentiment dataframe (mean sentiment per day)


Feature Engineering-

1-Close-Adjusted closing price of AAPL
2-Volume-Trading volume
3-Sentiment-Daily average compound sentiment score (VADER)
4-ema_20-20-day Exponential Moving Average
5-ema_50-50-day Exponential Moving AveragereturnDaily percentage 
6-return -(pct_change)
7-rsi_14-14-period Relative Strength Index
All 7 features are normalized using MinMaxScaler before being fed into the LSTM models.

Sliding Window Approach-

Window size: 60 trading days
For each window of 60 days, the model predicts the 61st day's closing price
Total samples: 1,198
Train/Test split: 80/20 → Train: 958 samples | Test: 240 samples


Model 1 — LSTM Regression (Price Prediction)
Architecture
LSTM(128, return_sequences=True)  →  Dropout(0.2)
LSTM(64,  return_sequences=True)  →  Dropout(0.1)
LSTM(64,  return_sequences=True)  →  Dropout(0.1)
LSTM(32)
Dense(32, activation='relu')
Dense(1,  activation='linear')

Training Config
Loss Function-Mean Squared Error (MSE)
Optimizer-Adam
Epochs-30 (with early stopping)
Batch Size-32
Validation Split-20%
Early Stopping-patience=20, restore best weights

Results-
MSE (on scaled predictions)~0.00059
MSE (on actual prices in USD)~4.85 (USD²)
RMSE (approx.)~$2.20

MSE values were computed both in the MinMax-scaled space and after inverse transforming back to actual USD prices.


Model 2 — LSTM Classifier (Direction Prediction)

Predicts whether the next day's close will be higher (1) or lower (0) than the current day's close.
Architecture
LSTM(128, return_sequences=True)  →  Dropout(0.2)
LSTM(128, return_sequences=True)  →  Dropout(0.2)
LSTM(64)
Dense(32, activation='relu')
Dense(1,  activation='sigmoid')

Training Config
Loss Function-Binary Cross-Entropy
Optimizer-Adam
Epochs-30 (with early stopping)
Batch Size-32
Validation Split-20%

Results-
MetricValueClassification Accuracy~54–57%
Threshold for classification: predicted probability > 0.5 → UP (1), else DOWN (0)


Monte Carlo Dropout (Uncertainty Estimation)-
To estimate prediction uncertainty, MC Dropout is applied at inference time:

The model is run 50 times with dropout active (training=True)
Mean prediction and standard deviation are computed across runs
A 95% Confidence Interval is plotted: mean ± 1.96 × std

This gives a probabilistic view of the predictions rather than just a point estimate.

Backtesting — Simple Trading Strategy
A simple long-only strategy was backtested using the regression model's actual price outputs:

Buy signal: If y_actual[i+1] > y_actual[i] (next day price is higher)
Sell/hold signal: Otherwise
Initial Capital: $100,000

Strategy vs Buy & Hold
The portfolio value over the test period is plotted against a baseline Buy & Hold strategy to evaluate real-world usefulness.

Tech Stack-
Pandas-Data manipulation
Numpy-Numerical operations
yfinance-Downloading stock price data
nltk (VADER)Sentiment analysis on news headlines
scikit-learn-MinMaxScaler, MSE, accuracy_score
tensorflow/keras-LSTM model building & training
matplotlib-Visualizations

How to Run-
1. Clone the Repository
bashgit clone https://github.com/your-username/stock-market-trend.git
cd stock-market-trend
2. Install Dependencies
bashpip install pandas numpy yfinance nltk scikit-learn tensorflow matplotlib
3. Download NLTK Data
pythonimport nltk
nltk.download('vader_lexicon')
4. Add the News Dataset
Place archive (3).zip (containing cnbc_headlines.csv, guardian_headlines.csv, reuters_headlines.csv) in the project root directory.
5. Run the Notebook
bashjupyter notebook stock_market_trend.ipynb

Note: You can also try other tickers. Supported tickers include:
AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA, GS, KO, PEP etc


Sample Visualizations-

Training vs Validation Loss — MSE loss curves over epochs for both models
Actual vs Predicted Close Price — Line chart comparing ground truth and LSTM predictions on test data
MC Dropout with 95% Confidence Interval — Uncertainty band around predictions
Portfolio Backtest — ML Strategy vs Buy & Hold comparison


Future Improvements-

 Add Transformer-based model (e.g., Temporal Fusion Transformer)
 Include more tickers for portfolio-level predictions
 Use a FinBERT model instead of VADER for more accurate financial sentiment
 Deploy as a real-time dashboard using Streamlit or Dash
 Add more advanced backtesting metrics (Sharpe ratio, max drawdown)
