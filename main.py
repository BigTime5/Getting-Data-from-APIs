# main.py
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from config import settings
from data import SQLRepository, AlphaVantage
from model import GarchModel

# Initialize FastAPI app
app = FastAPI(
    title="Stock Volatility API",
    description="API for fitting GARCH models and forecasting stock volatility.",
    version="1.0.0"
)

# Global database connection
connection = sqlite3.connect(settings.db_name, check_same_thread=False)
repo = SQLRepository(connection=connection)

# Input and output models
class FitIn(BaseModel):
    ticker: str = Field(..., min_length=1, description="Stock ticker symbol (e.g., 'AMZN')")
    use_new_data: bool = Field(..., description="Whether to fetch new data from Alpha Vantage")
    n_observations: int = Field(..., gt=0, description="Number of observations for training")
    p: int = Field(..., ge=0, description="Lag order of symmetric innovation")
    q: int = Field(..., ge=0, description="Lag order of lagged volatility")

class FitOut(FitIn):
    success: bool
    message: str

class PredictIn(BaseModel):
    ticker: str = Field(..., min_length=1, description="Stock ticker symbol (e.g., 'AMZN')")
    n_days: int = Field(..., gt=0, description="Forecast horizon in days")

class PredictOut(PredictIn):
    success: bool
    forecast: dict
    message: str

def build_model(ticker: str, use_new_data: bool) -> GarchModel:
    """Build GarchModel instance with repository."""
    return GarchModel(ticker=ticker, use_new_data=use_new_data, repo=repo)

@app.get("/hello", status_code=200)
async def hello():
    """Return a greeting message."""
    return {"message": "Hello world!"}

@app.post("/fit", status_code=200, response_model=FitOut)
async def fit_model(request: FitIn):
    """Fit a GARCH model and save it to disk.

    Parameters
    ----------
    request : FitIn
        Input parameters for model fitting.

    Returns
    -------
    FitOut
        Confirmation of model fitting with success status and message.
    """
    try:
        model = build_model(ticker=request.ticker, use_new_data=request.use_new_data)
        model.wrangle_data(n_observations=request.n_observations)
        model.fit(p=request.p, q=request.q)
        filename = model.dump()
        return FitOut(
            **request.dict(),
            success=True,
            message=f"Trained and saved to '{filename}'."
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/predict", status_code=200, response_model=PredictOut)
async def get_prediction(request: PredictIn):
    """Generate volatility forecast using a saved GARCH model.

    Parameters
    ----------
    request : PredictIn
        Input parameters for prediction.

    Returns
    -------
    PredictOut
        Volatility forecast with success status and message.
    """
    try:
        model = build_model(ticker=request.ticker, use_new_data=False)
        model.load()
        prediction = model.predict_volatility(horizon=request.n_days)
        return PredictOut(
            **request.dict(),
            success=True,
            forecast=prediction,
            message=f"Forecast generated for {request.ticker} over {request.n_days} days."
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    connection.close()