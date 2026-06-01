Stock Market Trend Prediction using LSTM and News Sentiment Analysis
Overview of project-

This project predicts stock market trends by combining:

1-Historical stock price data
2-Technical indicators
3-Financial news sentiment analysis
4-Deep Learning (LSTM Networks)

The model uses Apple's (AAPL) stock data along with news headlines from multiple financial news sources to forecast future stock prices and classify market direction (up/down).

The goal is to improve stock prediction performance by incorporating both quantitative market data and qualitative market sentiment.

Features-

1-Financial Data Collection
Downloaded historical stock data using Yahoo Finance (yfinance)
Used Apple (AAPL) stock prices from 2020–2025

2-News Sentiment Analysis
Integrated financial headlines from:
CNBC
Reuters
The Guardian
Applied VADER Sentiment Analysis to calculate sentiment scores for each news article.

3-Feature Engineering
Generated several technical indicators:
Daily Returns
20-Day EMA
50-Day EMA
RSI (Relative Strength Index)
Trading Volume
News Sentiment Score

4-Deep Learning Models

1-Stock Price Regression Model
Predicts future closing prices using:

Multi-layer LSTM architecture
Dropout Regularization
Mean Squared Error Loss

2-Trend Classification Model
Predicts whether the next day's price will:

Increase (1)
Decrease (0)
using:

LSTM Network
Sigmoid Output Layer
Binary Cross Entropy Loss

5-Uncertainty Estimation
Implemented:

Monte Carlo Dropout
95% Confidence Intervals
to estimate prediction uncertainty.

6-Strategy Backtesting

Compared:

ML-based trading strategy
Buy-and-Hold strategy

using simulated portfolio returns.

Dataset-

Stock Data Source:
Yahoo Finance
Ticker Used:
AAPL (Apple Inc.)

News Data
Financial headlines collected from:

CNBC
Reuters
The Guardian

Project Workflow

Stock Price Data
        │
Technical Indicators
        │
News Headlines
        │
Sentiment Analysis (VADER)
        │
Feature Engineering
        │
Data Scaling
        │
Sequence Creation (60-Day Window)
        │
LSTM Models
    ┌───────┴────────┐ 
Price Forecast    Trend Classification
    │                │
    └───────┬────────┘
Trading Strategy Evaluation

Technologies Used-

1-Programming Language-Python

2-Data Processing
Pandas
NumPy

3-Data Collection
yFinance
NLP
NLTK
VADER Sentiment Analyzer

4-Machine Learning / Deep Learning
TensorFlow
Keras
Scikit-Learn

5-Visualization
Matplotlib

Model Architecture-

Price Prediction Model
LSTM (128)
↓
Dropout (0.2)
↓
LSTM (64)
↓
Dropout (0.1)
↓
LSTM (64)
↓
Dropout (0.1)
↓
LSTM (32)
↓
Dense (32)
↓
Dense (1)

Trend Classification Model-

LSTM (128)
↓
Dropout (0.2)
↓
LSTM (128)
↓
Dropout (0.2)
↓
LSTM (64)
↓
Dense (32)
↓
Dense (1, Sigmoid)

Results-
The project evaluates:

Regression Metrics
Mean Squared Error (MSE)
Actual vs Predicted Closing Prices
Classification Metrics
Accuracy Score
Risk Estimation
Monte Carlo Dropout
Confidence Intervals
Trading Performance
ML Strategy Portfolio Growth
Buy & Hold Benchmark
Visualizations

The notebook includes-

Training Loss Curves
Validation Loss Curves
Actual vs Predicted Stock Prices
Confidence Interval Plots
Classification Accuracy Curves
Strategy Backtesting Results
Future Improvements
Incorporate additional stocks
Use Transformer-based models
Integrate real-time news feeds
Add macroeconomic indicators
Deploy as a web application
Use FinBERT instead of VADER for financial sentiment analysis

Key Learnings-

Time Series Forecasting
LSTM Networks
Financial Sentiment Analysis
Feature Engineering
Model Uncertainty Estimation
Quantitative Trading Strategy Evaluation
Deep Learning for Financial Markets
