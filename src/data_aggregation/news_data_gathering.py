
class NewsData:




class StockDataPull_AlphaVantage:
    """
    Class to encapsulate stock data pulling functionality.

    Attributes:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for data retrieval ('YYYY-MM-DD').
        end_date (str): End date for data retrieval ('YYYY-MM-DD').
        api_key (str): API key for Alpha Vantage.
    """

    def __init__(self, ticker, start_date, end_date):
        """
        Initialize StockDataPull_AlphaVantage with a ticker, start_date, and end_date.

        Parameters:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date for data retrieval ('YYYY-MM-DD').
            end_date (str): End date for data retrieval ('YYYY-MM-DD').
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.setup()

    def setup(self):


    def date_range(self):
        """
        Optimized method to create a list of dates that are used to retrieve data from Alpha Vantage.
        """
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
        date_range = pd.date_range(start_date, end_date, freq='M')  # Monthly frequency
        date_list = [f"month={date.strftime('%Y-%m')}" for date in date_range]
        return date_list

    def get_alphavantage_pricing_data(self):
        headers = {'Content-Type': "application/json", 'Authorization': f"Token {self.api_key}"}
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={self.ticker}&interval=1min&outputsize=full&apikey={self.api_key}'
        r = requests.get(url, headers=headers)
        data = r.json()
        time_series_data = data.get('Time Series (1min)', {})
        df = pd.DataFrame.from_dict(time_series_data, orient='index')
        df.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low', 
            '4. close': 'Close', '5. volume': 'Volume'}, inplace=True)
        df['Symbol'] = self.ticker
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date/Time'}, inplace=True)
        return df[['Symbol', 'Date/Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def get_alphavantage_news_data(self):
        news_start_date = f"{self.start_date.replace('-', '')}T0000"
        news_end_date = f"{self.end_date.replace('-', '')}T0000"
        response = requests.get(f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.ticker}&limit=1000&time_from={news_start_date}&time_to={news_end_date}&apikey={self.api_key}')
        data = json.loads(response.text)
        news_df = pd.DataFrame(data['feed'])
        news_df['Symbol'] = self.ticker
        news_df['time_published'] = pd.to_datetime(news_df['time_published'], format='%Y%m%dT%H%M%S')
        return news_df

    def merge_pricing_and_news_data(self, pricing_df, news_df):
        """
        Append pricing and news data based on 'Date/Time'.

        Parameters:
            pricing_df (DataFrame): The pricing data DataFrame with 'Date/Time' column.
            news_df (DataFrame): The news data DataFrame with 'time_published' column.

        Returns:
            DataFrame: A DataFrame combining both pricing and news data.
        """
        # Rename 'time_published' in news_df to 'Date/Time' to match pricing_df
        news_df.rename(columns={'time_published': 'Date/Time'}, inplace=True)

        # Ensure both 'Date/Time' columns are in datetime64[ns] format
        pricing_df['Date/Time'] = pd.to_datetime(pricing_df['Date/Time'])
        news_df['Date/Time'] = pd.to_datetime(news_df['Date/Time'])

        # Perform an outer join on the 'Date/Time' column
        merged_df = pd.merge(pricing_df, news_df, on='Date/Time', how='outer', suffixes=('_pricing', '_news'))
        return merged_df

    def run(self):
        """
        Orchestrate data retrieval, processing, and merging tasks.

        Returns:
            DataFrame: Final merged DataFrame containing both pricing and news data.
        """
        alphavantage_pricing_df = self.get_alphavantage_pricing_data()
        alphavantage_news_df = self.get_alphavantage_news_data()

        # Convert 'Date/Time' in pricing_df and 'time_published' in news_df to datetime
        alphavantage_pricing_df['Date/Time'] = pd.to_datetime(alphavantage_pricing_df['Date/Time'])
        alphavantage_news_df['time_published'] = pd.to_datetime(alphavantage_news_df['time_published'])

        # Remove seconds from 'Date/Time' and 'time_published'
        alphavantage_pricing_df['Date/Time'] = alphavantage_pricing_df['Date/Time'].dt.floor('min')
        alphavantage_news_df['time_published'] = alphavantage_news_df['time_published'].dt.floor('min')

        # Merge the pricing and news data on 'Date/Time' and 'time_published'
        merged_df = self.merge_pricing_and_news_data(alphavantage_pricing_df, alphavantage_news_df)

        # Ensure the DataFrame is sorted by 'Date/Time' to correctly apply forward fill
        merged_df.sort_values(by='Date/Time', inplace=True)

        # Forward fill NaN values in the 'Close' column (or any other stock price related columns as needed)
        columns_to_fill = ['Open', 'High', 'Low', 'Close', 'Volume']
        merged_df[columns_to_fill] = merged_df[columns_to_fill].ffill()

        # Return all three DataFrames
        return alphavantage_pricing_df, alphavantage_news_df, merged_df