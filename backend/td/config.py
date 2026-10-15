"""Central configuration, read from environment / .env.

Every process (ingestor, analytics, gateway, backfill) imports `settings` from here,
so one .env file configures the whole system.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), env_prefix="", extra="ignore")

    # ---- data source ------------------------------------------------------
    # "sim"  = built-in market simulator (no credentials needed)
    # "dhan" = live Dhan Data API
    data_source: Literal["sim", "dhan"] = Field("sim", alias="DATA_SOURCE")

    # Dhan credentials (only needed when DATA_SOURCE=dhan or for backfill)
    dhan_client_id: str = Field("", alias="DHAN_CLIENT_ID")
    dhan_access_token: str = Field("", alias="DHAN_ACCESS_TOKEN")
    # "basic" = ₹499 Data API plan, "pro" = ₹999 Pro Traders plan
    # pro unlocks: news headlines, market movers, CAS feed, 200-level depth
    dhan_plan: Literal["basic", "pro"] = Field("pro", alias="DHAN_PLAN")
    dhan_base_url: str = Field("https://api.dhan.co/v2", alias="DHAN_BASE_URL")
    dhan_feed_url: str = Field("wss://api-feed.dhan.co", alias="DHAN_FEED_URL")
    dhan_depth20_url: str = Field("wss://depth-api-feed.dhan.co/twentydepth", alias="DHAN_DEPTH20_URL")
    dhan_scrip_master_url: str = Field(
        "https://images.dhan.co/api-data/api-scrip-master-detailed.csv", alias="DHAN_SCRIP_MASTER_URL"
    )

    # Dhan published limits (Oct 2026 docs) — keep configurable in case they change
    limit_data_per_sec: float = Field(5, alias="LIMIT_DATA_PER_SEC")
    limit_data_per_day: int = Field(7000, alias="LIMIT_DATA_PER_DAY")
    limit_quote_per_sec: float = Field(1, alias="LIMIT_QUOTE_PER_SEC")
    limit_chain_unique_per_8s: int = Field(10, alias="LIMIT_CHAIN_UNIQUE_PER_8S")

    # ---- infrastructure ---------------------------------------------------
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    # tick stream is transport only (analytics reads it live; the recorder keeps history on disk),
    # so a short buffer is enough: 1500 batches x 100 ms = 2.5 min of replay for a restarted worker
    stream_maxlen: int = Field(1500, alias="STREAM_MAXLEN")
    data_dir: Path = Field(ROOT / "data", alias="DATA_DIR")

    # ---- automatic history (live mode) --------------------------------------
    # On start-up and every morning the ingestor pulls what the live desks need from Dhan's
    # historical API: ~1 year of daily bars, 30 days of 1-minute bars, today's bars so far and
    # yesterday's futures close/OI (~850 data calls/day of the 7,000 budget). The full 5-year
    # history is still backfill/backfill.py's job.
    auto_history: bool = Field(True, alias="AUTO_HISTORY")
    auto_history_days: int = Field(400, alias="AUTO_HISTORY_DAYS")

    # ---- universe ---------------------------------------------------------
    # "fno" = all F&O stocks (+ near-month futures + indices). "custom" = SYMBOLS list
    universe: Literal["fno", "custom"] = Field("fno", alias="UNIVERSE")
    symbols: str = Field("", alias="SYMBOLS")
    chain_underlyings: str = Field("NIFTY,BANKNIFTY,FINNIFTY", alias="CHAIN_UNDERLYINGS")

    # ---- simulator --------------------------------------------------------
    sim_speed: float = Field(1.0, alias="SIM_SPEED")          # 1.0 = real time, 10 = 10x faster
    sim_start: str = Field("09:13:30", alias="SIM_START")     # simulated IST start time
    sim_seed: int = Field(0, alias="SIM_SEED")               # 0 = new random day each start
    sim_symbols: int = Field(180, alias="SIM_SYMBOLS")

    # ---- strategy ---------------------------------------------------------
    top_n: int = Field(5, alias="TOP_N")
    lock_time: str = Field("09:30:00", alias="LOCK_TIME")     # IST time the intraday picks lock
    or_minutes: int = Field(15, alias="OR_MINUTES")           # opening range length
    min_turnover_cr: float = Field(5.0, alias="MIN_TURNOVER_CR")  # liquidity filter (₹ crore traded by lock)

    # ---- GEO desk (world events) ------------------------------------------
    # auto = simulated world feeds when DATA_SOURCE=sim, real free feeds when DATA_SOURCE=dhan
    geo_enabled: bool = Field(True, alias="GEO_ENABLED")
    geo_source: Literal["auto", "live", "sim"] = Field("auto", alias="GEO_SOURCE")
    # comma list of layers the host never polls: ships,air,conflict,fires,chokepoints,risk,macro
    geo_disable: str = Field("", alias="GEO_DISABLE")
    aisstream_api_key: str = Field("", alias="AISSTREAM_API_KEY")   # free key: aisstream.io (live ships)
    firms_map_key: str = Field("", alias="FIRMS_MAP_KEY")           # free key: firms.modaps.eosdis.nasa.gov
    fred_api_key: str = Field("", alias="FRED_API_KEY")             # optional; CSV download works without
    adsb_api_url: str = Field("https://api.adsb.lol", alias="ADSB_API_URL")  # or https://api.airplanes.live
    geo_gdelt_backfill_hours: int = Field(12, alias="GEO_GDELT_BACKFILL_HOURS")
    geo_to_news: bool = Field(False, alias="GEO_TO_NEWS")           # push high-severity geo signals into NEWS
    # live MCX / currency futures shown on the GEO desk (Dhan feed): near-month contracts
    geo_macro_symbols: str = Field("CRUDEOIL,NATURALGAS,GOLD,SILVER,USDINR", alias="GEO_MACRO_SYMBOLS")

    # ---- gateway (used by run.py / docker) --------------------------------
    gateway_host: str = Field("0.0.0.0", alias="GATEWAY_HOST")
    gateway_port: int = Field(8000, alias="GATEWAY_PORT")
    # comma separated name:key pairs for friends, e.g. "mayank:k1,rahul:k2". Empty = open access.
    access_keys: str = Field("", alias="ACCESS_KEYS")
    cors_origins: str = Field("*", alias="CORS_ORIGINS")
    # Accounts (login page, /admin): the first admin is created from these if it does not exist yet. Without
    # them, /login offers a one-time setup to the host itself (localhost) while there are no users.
    admin_user: str = Field("", alias="ADMIN_USER")
    admin_password: str = Field("", alias="ADMIN_PASSWORD")
    session_days: float = Field(30, alias="SESSION_DAYS")            # login lasts this long
    session_secret: str = Field("", alias="SESSION_SECRET")          # default: data/auth/secret.key (generated)

    # ---- research desks (AI analyst, Screener, Monte Carlo, Quant) ----------
    # AI analyst: "anthropic" (Claude, official SDK) or "openai" = any OpenAI-compatible chat-completions API
    # (OpenAI, OpenRouter, Groq, Together, DeepSeek, Gemini's OpenAI endpoint, a local Ollama/vLLM, ...).
    # No key = the AI desk still shows the data dossier and a rule-based view, without the written analysis.
    ai_provider: Literal["anthropic", "openai"] = Field("anthropic", alias="AI_PROVIDER")
    ai_api_key: str = Field("", alias="AI_API_KEY")       # falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY
    ai_model: str = Field("", alias="AI_MODEL")           # default claude-opus-5-5 (anthropic); required for openai
    ai_base_url: str = Field("", alias="AI_BASE_URL")     # openai: default https://api.openai.com/v1
    ai_effort: str = Field("high", alias="AI_EFFORT")     # anthropic effort: low|medium|high|xhigh|max ("" = omit)
    ai_max_tokens: int = Field(64000, alias="AI_MAX_TOKENS")   # thinking + the JSON report; streamed, so a cap only
    ai_timeout: float = Field(600, alias="AI_TIMEOUT")    # seconds per model call
    ai_daily_limit: int = Field(40, alias="AI_DAILY_LIMIT")   # AI analyses per day for the whole install
    ai_user_daily_limit: int = Field(10, alias="AI_USER_DAILY_LIMIT")   # per user (admins are exempt)
    # Screener.in fundamentals: cached on disk, fetched politely (one page at a time, never search URLs)
    screener_ttl_hours: float = Field(12, alias="SCREENER_TTL_HOURS")
    screener_min_interval: float = Field(3.0, alias="SCREENER_MIN_INTERVAL")   # seconds between page fetches
    # market assumptions for Monte Carlo drift (CAPM) — annual, decimal
    risk_free_rate: float = Field(0.065, alias="RISK_FREE_RATE")       # ~India 10y G-sec
    equity_risk_premium: float = Field(0.06, alias="EQUITY_RISK_PREMIUM")
    research_fetch_reserve: int = Field(500, alias="RESEARCH_FETCH_RESERVE")   # keep this many Dhan calls/day

    @property
    def lake_dir(self) -> Path:
        sub = "sim" if self.data_source == "sim" else "dhan"
        return Path(self.data_dir) / "lake" / sub

    @property
    def recorder_dir(self) -> Path:
        return Path(self.data_dir) / "recorder"

    @property
    def is_pro(self) -> bool:
        return self.dhan_plan == "pro"

    @property
    def geo_mode(self) -> str:
        if self.geo_source != "auto":
            return self.geo_source
        return "sim" if self.data_source == "sim" else "live"

    def geo_disabled(self) -> set[str]:
        return {x.strip().lower() for x in self.geo_disable.split(",") if x.strip()}

    def access_key_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for pair in self.access_keys.split(","):
            pair = pair.strip()
            if ":" in pair:
                name, key = pair.split(":", 1)
                out[key.strip()] = name.strip()
        return out

    def chain_list(self) -> list[str]:
        return [s.strip().upper() for s in self.chain_underlyings.split(",") if s.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
