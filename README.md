Finance App - Stock Market Analysis and Prediction

Overview
This is a Django-based web application designed to provide a comprehensive platform for stock market analysis. It integrates financial indicators, news sentiment analysis, macroeconomic data, and machine learning models to give users detailed insights into market trends and potential investment opportunities.

Features

Stock Data and News: Fetches stock data from Yahoo Finance and provides up-to-date news and sentiment analysis.
Technical Indicators: Calculates popular financial indicators such as SMA (Simple Moving Average), RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), Bollinger Bands, and ATR (Average True Range) for stock analysis.

Macroeconomic Analysis: Integrates macroeconomic data from the Federal Reserve Economic Data (FRED) to provide insights into GDP, inflation rates, unemployment, interest rates, and consumer confidence.

Sentiment Analysis: Analyzes news articles using TextBlob and VaderSentiment to determine sentiment polarity (positive, negative, neutral).

Predictive Modeling: Implements stock price prediction using ARIMA, Linear Regression, and LSTM (Long Short-Term Memory) models.

Portfolio Management: Tracks the user's portfolio performance in real-time and updates hourly.

Interactive Charts: Visualizes stock data, market indicators, and predictions using Plotly and Matplotlib.


Technologies Used

Backend:
Django: A high-level Python web framework for rapid development and clean, pragmatic design.
Celery: For asynchronous task processing, used to handle tasks like updating portfolio performance or fetching large datasets.
Statsmodels: For time series forecasting, particularly using ARIMA models.
Keras: For implementing LSTM-based machine learning models to predict stock prices.
Scikit-learn: For data preprocessing (e.g., MinMaxScaler) and implementing linear regression models.
Pandas: For data manipulation and analysis.
NumPy: For numerical computing and working with arrays.
TextBlob: For basic sentiment analysis.
VaderSentiment: A more specialized sentiment analysis tool for analyzing financial news articles.
Requests: For making HTTP requests to external APIs, such as the News API.
BeautifulSoup: For web scraping and parsing HTML to extract relevant news data.

Frontend:
Open-source **[Django Template](https://www.creative-tim.com/templates/django)** crafted on top of **Black Dashboard**, a 
Plotly: For creating interactive and visually appealing charts for financial data.
Matplotlib: For static charts, used alongside Seaborn for more complex visualizations.
Django Templates: To render the HTML pages dynamically with data passed from the backend.

Database:
PostgreSQL: A powerful, open-source object-relational database system for storing portfolio data and stock market indicators.
Django ORM: For interacting with the database and managing models.

API Integration:
Yahoo Finance API: Used to fetch historical and real-time stock data.
FRED API: Provides access to macroeconomic data like GDP, CPI, unemployment rate, etc.
News API: Fetches news articles for sentiment analysis.

Setup
Prerequisites
Python 3.8 or higher
Django 3.2 or higher
PostgreSQL (for database setup)
Virtual environment (recommended for project isolation)
Installation Steps


Clone the repository:

git clone https://github.com/yourusername/finance_app.git
cd finance_app

Create a virtual environment:

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
pip install -r requirements.txt

Set up the database (PostgreSQL):

Create a new database (e.g., finance_app).
Update the DATABASES settings in settings.py with your database credentials.

Apply migrations to create the database schema: 
python manage.py migrate
Create a superuser to access the Django admin panel:
bash
 
python manage.py createsuperuser
Run the development server:
bash
 
python manage.py runserver
Visit http://127.0.0.1:8000 in your browser to access the application.

Application Flow
Dashboard (index view)
The main page of the app fetches real-time stock data, calculates technical indicators, performs macroeconomic analysis, and shows sentiment analysis from the latest news. The data is then rendered in the dashboard with interactive charts.

Sentiment Analysis
The app fetches news related to the selected stock symbol using the News API. Sentiment analysis is performed using TextBlob and VaderSentiment. The sentiment scores (positive, neutral, negative) are displayed alongside news articles.

Financial Indicators
Technical indicators such as Moving Averages, RSI, MACD, Bollinger Bands, and ATR are calculated using historical stock data and displayed to the user to assist with decision-making.

Predictive Models
Stock price predictions are made using ARIMA and LSTM models. These models predict future price movements based on historical data and display the results in an interactive chart.

Macroeconomic Analysis
Economic indicators like GDP, CPI, PPI, unemployment rate, and consumer confidence are fetched from the FRED API. The app analyzes these indicators and provides a macroeconomic suggestion based on the current data.

Portfolio Management
Users can track their portfolios, with the app updating performance in real-time. Portfolio data is fetched and updated hourly.

Example of Key Functions
fetch_news_sentiment(symbol)
Fetches stock-related news and analyzes the sentiment of the titles using TextBlob and VaderSentiment. Returns the dominant sentiment (positive, negative, neutral).

calculate_indicators(df)
Calculates various financial indicators (SMA, RSI, MACD, Bollinger Bands, ATR) from the stock's historical data stored in a Pandas DataFrame.

analyze_macroeconomic_data()
Fetches macroeconomic data from FRED (GDP, CPI, unemployment rate, etc.) and returns analysis with suggestions based on the data.

predict_stock_price()
Uses ARIMA and LSTM models to predict future stock prices based on historical data. The predictions are visualized using Plotly.

Running Background Tasks with Celery
The app uses Celery to handle background tasks like updating portfolio performance and fetching stock data at regular intervals. This ensures the main application remains responsive while heavy computations are offloaded.


Future Enhancements
Real-time data updates: Implement WebSocket-based updates for real-time stock data.
Extended machine learning models: Incorporate additional models like XGBoost, Prophet, or Reinforcement Learning for stock price predictions.
User authentication and portfolio management: Extend user authentication and allow users to manage their stock portfolios more effectively.
Mobile app support: Provide a mobile-friendly version or create a mobile app for easier access to stock analysis.
Contributions
Feel free to fork the repository and submit pull requests. Contributions in the form of bug fixes, new features, or improvements are always welcome.
