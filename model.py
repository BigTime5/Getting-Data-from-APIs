# model.py
import os
from glob import glob
import joblib
import pandas as pd
from arch import arch_model
from config import settings
from data import AlphaVantage, SQLRepository

class GarchModel:
    """Class for training GARCH model and generating volatility predictions.
    
    Attributes
    -----------
    ticker : str
        Ticker symbol of the equity whose volatility will be predicted.
    repo : SQLRepository
        Repository where training data is stored.
    use_new_data : bool
        Whether to download new data from AlphaVantage or use existing data.
    model_directory : str
        Path to directory where trained models are stored.
    data : pd.Series
        Equity returns for training (set by wrangle_data).
    model : arch.univariate.base.ARCHModelResult
        Trained GARCH model (set by fit).
    aic : float
        Akaike Information Criterion of the trained model (set by fit).
    bic : float
        Bayesian Information Criterion of the trained model (set by fit).
    
    Methods
    --------
    wrangle_data
        Generate equity returns from database or API.
    fit
        Fit GARCH model to training data.
    predict_volatility
        Generate volatility forecast from trained model.
    dump
        Save trained model to file.
    load
        Load most recent trained model from file.
    """
    def __init__(self, ticker: str, repo: SQLRepository, use_new_data: bool):
        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data
        self.model_directory = settings.model_directory
        os.makedirs(self.model_directory, exist_ok=True)  # Ensure directory exists

    def wrangle_data(self, n_observations: int) -> pd.Series:
        """Extract data from database (or AlphaVantage), transform for training, and attach to self.data.
        
        Parameters
        -----------
        n_observations : int
            Number of observations to retrieve.
        
        Returns
        --------
        pd.Series
            Equity returns (decimal form, no NaN values).
        """
        if not isinstance(n_observations, int) or n_observations <= 0:
            raise ValueError("n_observations must be a positive integer")

        # Add new data if required
        if self.use_new_data:
            api = AlphaVantage()
            new_data = api.get_daily(ticker=self.ticker)
            self.repo.insert_table(
                table_name=self.ticker, records=new_data, if_exists="replace"
            )

        # Pull data from database
        df = self.repo.read_table(table_name=self.ticker, limit=n_observations + 1)
        
        # Calculate returns
        df.sort_index(ascending=True, inplace=True)
        df["return"] = df["close"].pct_change()  # Decimal returns for GARCH
        self.data = df["return"].dropna().rename("return")
        
        if len(self.data) < n_observations:
            raise ValueError(f"Requested {n_observations} returns, but only {len(self.data)} available")
        
        return self.data

    def fit(self, p: int, q: int) -> None:
        """Fit GARCH model to self.data and attach to self.model. Assigns AIC and BIC.
        
        Parameters
        -----------
        p : int
            Lag order of the symmetric innovation.
        q : int
            Lag order of lagged volatility.
        
        Returns
        --------
        None
        """
        if not isinstance(p, int) or p < 0:
            raise ValueError("p must be a non-negative integer")
        if not isinstance(q, int) or q < 0:
            raise ValueError("q must be a non-negative integer")
        if not hasattr(self, 'data') or self.data.empty:
            raise ValueError("No data available. Run wrangle_data first.")
        
        # Train model
        self.model = arch_model(self.data, p=p, q=q, rescale=False).fit(disp=0)
        self.aic = self.model.aic
        self.bic = self.model.bic

    def __clean_prediction(self, prediction: pd.DataFrame) -> dict:
        """Reformat model prediction to JSON.
        
        Parameters
        -----------
        prediction : pd.DataFrame
            Variance from an ARCH model forecast.
        
        Returns
        -----------
        dict
            Forecast of volatility. Keys are dates in ISO 8601 format, values are volatility.
        """
        if not isinstance(prediction, pd.DataFrame):
            raise ValueError("Prediction must be a pandas DataFrame")
        start = prediction.index[0] + pd.DateOffset(days=1)
        prediction_dates = pd.bdate_range(start=start, periods=prediction.shape[1])
        prediction_index = [d.isoformat() for d in prediction_dates]
        data = prediction.values.flatten() ** 0.5
        prediction_formatted = pd.Series(data.round(3), index=prediction_index)
        return prediction_formatted.to_dict()

    def predict_volatility(self, horizon: int = 5) -> dict:
        """Predict volatility using self.model.
        
        Parameters
        -----------
        horizon : int
            Forecast horizon (default: 5).
        
        Returns
        -----------
        dict
            Volatility forecasts. Keys are dates in ISO 8601 format, values are volatility.
        """
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if not hasattr(self, 'model'):
            raise ValueError("No model available. Run fit first.")
        
        prediction = self.model.forecast(horizon=horizon, reindex=False).variance
        return self.__clean_prediction(prediction)

    def dump(self) -> str:
        """Save model to self.model_directory with timestamp.
        
        Returns
        -----------
        str
            Filepath where model was saved.
        """
        if not hasattr(self, 'model'):
            raise ValueError("No model to save. Run fit first.")
        
        timestamp = pd.Timestamp.now().isoformat().replace(':', '-')  # Safe filename
        filepath = os.path.join(self.model_directory, f"{timestamp}_{self.ticker}.pkl")
        joblib.dump(self.model, filepath)
        return filepath

    def load(self) -> None:
        """Load most recent model in self.model_directory for self.ticker, attach to self.model."""
        pattern = os.path.join(self.model_directory, f"*{self.ticker}.pkl")
        try:
            model_path = sorted(glob(pattern))[-1]
            self.model = joblib.load(model_path)
        except IndexError:
            raise FileNotFoundError(f"No model found for '{self.ticker}' in {self.model_directory}")