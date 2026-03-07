# Twitter Engagement Analysis

A comprehensive social network analysis project that analyzes Twitter engagement patterns using machine learning, deep learning, and time-series forecasting.

## 📊 Project Overview

This project performs end-to-end analysis of Twitter data to predict and optimize user engagement. It includes data preprocessing, feature engineering, multiple ML models, and visualizations to derive actionable insights.

## 🚀 Features

- **Data Preprocessing**: Clean and prepare raw Twitter data
- **Feature Engineering**: Create meaningful features from tweet metadata
- **Machine Learning**: XGBoost regression for engagement prediction
- **Statistical Analysis**: StatsModels OLS regression
- **Deep Learning**: PyTorch neural network for engagement scoring
- **Time-Series Forecasting**: Prophet for engagement trend prediction
- **Visualizations**: Comprehensive charts and graphs
- **User Analytics**: Best posting times, top users, engagement metrics

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities
- **XGBoost** - Gradient boosting for regression
- **Statsmodels** - Statistical modeling
- **Prophet** - Time-series forecasting
- **PyTorch** - Deep learning framework
- **Matplotlib & Seaborn** - Data visualization

## 📁 Project Structure

```
Social Network/
├── code.ipynb                 # Main analysis notebook
├── Twitter_Project_Documentation (1).pdf  # Project documentation
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## 📦 Installation

Install required dependencies:

```bash
pip install pandas numpy scikit-learn xgboost statsmodels prophet torch matplotlib seaborn
```

## 🔧 Usage

1. Ensure you have a `Twitterdatainsheets.csv` file in the project directory
2. Run the notebook:

```bash
jupyter notebook code.ipynb
```

Or use any Python environment to execute the code.

## 📈 Key Outputs

The project generates several output files:
- `tweets_clean.csv` - Cleaned dataset
- `tweets_features.csv` - Engineered features
- `user_engagement_summary.csv` - User-level analytics
- `user_best_hour_summary.csv` - Optimal posting times
- `tweet_engagement_summary.csv` - Top tweets analysis

## 🔍 Analysis Highlights

- **Engagement Score Calculation**: `Likes × 1.2 + Retweets × 1.5`
- **Features Used**: Tweet length, hashtags, mentions, URLs, posting hour
- **Best Posting Hour**: Identified for maximizing engagement
- **User Insights**: Top contributors and engagement patterns

## 🤖 Models

| Model | Purpose |
|-------|---------|
| XGBoost Regressor | Predict engagement scores |
| StatsModels OLS | Statistical significance testing |
| PyTorch Neural Network | Deep learning engagement prediction |
| Prophet | Time-series forecasting |

## 📝 License

This project is for educational purposes.

## 👤 Author

Created as part of Social Network analysis project.

---

*Generated with ❤️ using Python and Machine Learning*

