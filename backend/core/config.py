from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    binance_api_key: str = ""
    binance_api_secret: str = ""
    environment: str = "development"
    # SQLite default for prototype; override with PostgreSQL URL in production
    database_url: str = "sqlite:///./database/trading.db"

    # Pydantic v2 settings config
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
