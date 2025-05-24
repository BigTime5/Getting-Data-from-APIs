import pandas as pd
import requests
import sqlite3
import re
from urllib.parse import urlencode
import time
import logging
from config import settings

# Global logging configuration for data.py; override in main script if needed
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaVantage:
    def __init__(self, api_key: str = settings.alpha_api_key):
        """Initialize AlphaVantage API client.

        Args:
            api_key (str): Alpha Vantage API key. Defaults to settings.alpha_api_key.
        """
        self._api_key = api_key

    def get_daily(self, ticker: str, output_size: str = "full") -> pd.DataFrame:
        """Get daily time series of an equity from Alpha Vantage API.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'IBM').
            output_size (str): The size of the output ('compact' or 'full'). Defaults to 'full'.

        Returns:
            pd.DataFrame: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                          indexed by date.

        Raises:
            ValueError: If ticker is invalid, output_size is invalid, or API call fails.
        """
        # Validate inputs
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("Ticker must be a non-empty string")
        if output_size not in ["compact", "full"]:
            raise ValueError("output_size must be 'compact' or 'full'")

        # Construct URL
        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": output_size,
            "datatype": "json",
            "apikey": self._api_key
        }
        url = f"{base_url}?{urlencode(params)}"

        # Send request to API
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise for 4xx/5xx errors
            response_data = response.json()

            # Check for API errors
            if "Error Message" in response_data:
                raise ValueError(f"API Error: {response_data['Error Message']}")
            if "Note" in response_data:
                logger.info("Rate limit reached. Waiting 15 seconds...")
                time.sleep(15)
                return self.get_daily(ticker, output_size)  # Retry
            if "Time Series (Daily)" not in response_data:
                raise ValueError(f"Invalid API call. Check ticker symbol '{ticker}'")

            # Process data into DataFrame
            stock_data = response_data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(stock_data, orient="index")
            # Validate numeric data
            for col in ["1. open", "2. high", "3. low", "4. close", "5. volume"]:
                if not pd.to_numeric(df[col], errors="coerce").notnull().all():
                    raise ValueError(f"Non-numeric data found in column '{col}' for ticker '{ticker}'")
            df = df.astype({
                "1. open": "float64",
                "2. high": "float64",
                "3. low": "float64",
                "4. close": "float64",
                "5. volume": "int64"  # Use int64 to handle large volumes
            })
            df.columns = [c.split(". ")[1] if ". " in c else c for c in df.columns]
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            expected_columns = ["open", "high", "low", "close", "volume"]
            df = df[expected_columns]

            return df

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch data for {ticker}: {e}")

class SQLRepository:
    def __init__(self, connection: sqlite3.Connection):
        """Initialize SQLRepository with a SQLite connection.

        Args:
            connection (sqlite3.Connection): SQLite database connection.
        """
        self.connection = connection

    def insert_table(self, table_name: str, records: pd.DataFrame, if_exists: str = "fail") -> dict:
        """Insert DataFrame into SQLite database as a table.

        Args:
            table_name (str): Name of the table to insert into.
            records (pd.DataFrame): DataFrame containing stock data.
            if_exists (str): How to behave if the table already exists. Options are:
                - 'fail': Raise a ValueError
                - 'replace': Drop the table before inserting new values
                - 'append': Insert new values to the existing table
                Defaults to 'fail'.

        Returns:
            dict: Dictionary with keys:
                - 'transaction_successful': bool
                - 'records_inserted': int
                - 'error': str (only if transaction fails)

        Raises:
            ValueError: If table_name is invalid, records is not a DataFrame, or if_exists is invalid.
        """
        # Validate inputs
        if not isinstance(table_name, str) or not table_name:
            raise ValueError("Table name must be a non-empty string")
        if not re.match(r"^[a-zA-Z0-9_]+$", table_name):
            raise ValueError("Table name must contain only alphanumeric characters and underscores")
        if not isinstance(records, pd.DataFrame):
            raise ValueError("Records must be a pandas DataFrame")
        if if_exists not in ["fail", "replace", "append"]:
            raise ValueError("if_exists must be 'fail', 'replace', or 'append'")
        expected_columns = ["open", "high", "low", "close", "volume"]
        if not all(col in records.columns for col in expected_columns):
            raise ValueError(f"Records missing required columns: {expected_columns}")

        # Normalize table name (e.g., BRK.A -> BRK_A)
        table_name = table_name.replace(".", "_")

        try:
            n_inserted = records.to_sql(
                name=table_name, con=self.connection, if_exists=if_exists, index=True
            )
            return {
                "transaction_successful": True,
                "records_inserted": n_inserted
            }
        except Exception as e:
            return {
                "transaction_successful": False,
                "records_inserted": 0,
                "error": str(e)
            }

    def read_table(self, table_name: str, limit: int = None) -> pd.DataFrame:
        """Read table from SQLite database.

        Args:
            table_name (str): Name of the table in the SQLite database.
            limit (int, optional): Number of most recent records to retrieve (ordered by date descending).
                If None, all records are retrieved. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with DatetimeIndex 'date' and columns
                ['open', 'high', 'low', 'close', 'volume']. All columns are numeric.

        Raises:
            ValueError: If table_name is invalid, limit is invalid, or table read fails.
        """
        # Validate inputs
        if not isinstance(table_name, str) or not table_name:
            raise ValueError("Table name must be a non-empty string")
        if not re.match(r"^[a-zA-Z0-9_]+$", table_name):
            raise ValueError("Table name must contain only alphanumeric characters and underscores")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError("Limit must be a positive integer or None")

        # Normalize table name
        table_name = table_name.replace(".", "_")

        # Create SQL query
        if limit:
            sql = f"SELECT * FROM '{table_name}' ORDER BY date DESC LIMIT {limit}"
        else:
            sql = f"SELECT * FROM '{table_name}' ORDER BY date DESC"

        # Retrieve data
        try:
            df = pd.read_sql(
                sql=sql, con=self.connection, parse_dates=['date'], index_col="date"
            )
            # Validate columns
            expected_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in expected_columns):
                raise ValueError(f"Table '{table_name}' missing required columns: {expected_columns}")
            df = df[expected_columns].astype({
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "int64"
            })
            return df
        except Exception as e:
            raise ValueError(f"Failed to read table '{table_name}': {str(e)}")