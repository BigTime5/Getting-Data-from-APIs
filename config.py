"""Configuration module for the stock data analysis project.
Extracts environment variables from .env file for use across the application.
"""
import os
from pydantic_settings import BaseSettings

def return_full_path(filename: str = ".env") -> str:
    """Returns the absolute path to the .env file"""
    absolute_path = os.path.abspath(__file__)
    directory_name = os.path.dirname(absolute_path)
    full_path = os.path.join(directory_name, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The .env file was not found at: {full_path}")
    return full_path

class Settings(BaseSettings):
    """Uses pydantic to define settings for the project"""
    alpha_api_key: str
    db_name: str
    model_directory: str
    database_url: str

    model_config = {
        "protected_namespaces": ("settings_",),
        "env_file": return_full_path(".env"),
        "env_file_encoding": "utf-8"
    }

# Create instance of Settings for importing in other modules
settings = Settings()