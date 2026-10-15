# TradeDesk

A Bloomberg-style market terminal for NSE intraday, F&O and swing trading, built on the Dhan Data API.
**Data only: there is no order-placement code anywhere in this project.**

One machine connects to Dhan, does the analytics once and fans the results out to any number of
browsers (yours, your friends', your phone) through Redis. Adding a viewer costs no extra API calls.

```
F1 INTRADAY  09:30 top-5 long/short picks · live factor ranking · 20-level depth · factor breakdown · chart with SMC
F2 MARKET    breadth, A/D, PDH/PDL breaks · sector heatmap · gainers / losers / most active / volume shockers
F3 F&O       futures OI build-up (LB / SB / SC / LU) · option chain with PCR, max pain, GEX, gamma flip, walls, IV skew
F4 SWING     setup scanner (breakout, VCP, pullback, momentum, breakdown) · daily chart with EMA20/50 · setup receipts
F5 NEWS      catalyst feed classified by event, direction and novelty · catalyst leaderboard
F6 ALERTS    price / % / RVOL / VWAP / ORB / OI / news / top-list alerts · one-click alerts on the picks' stop & target
F7 GEO       world map: live ships, chokepoints & ports, war/conflict events, refinery fires, military aircraft,
             satellite imagery · geo-risk index · oil & supply · India macro · market impact — every view switchable

research desks — any NSE stock, from daily history + fundamentals (not the live feed):
F8 AI        analyst view for 6 months / 1 / 2 / 3 years from chart, fundamentals, quant model, Monte Carlo, market,
             news · any AI model from .env (Claude by default, or any OpenAI-compatible API) · rule-based without one
F9 SCREENER  Screener.in fundamentals: ratios, statements, growth, shareholding, peers · derived metrics · scorecard
F10 MONTE CARLO  10 000+ simulated price paths (GARCH-t, bootstrap, jumps, regimes, ensemble) · fan chart ·
             distribution · risk · target odds · calibration backtest that scores the bands against what happened
Alt+Q QUANT  next-day / week / month probabilities from a walk-forward-tested model (Kalman trend, GARCH, HMM
             regimes, momentum, mean reversion…) with its measured out-of-sample accuracy shown next to every number

/admin       a separate page for admins (log in at /login):
  USERS      accounts (admin / viewer), passwords, who is online
  HISTORY    pull the 5-year history from Dhan into the lake · progress, quota, coverage per symbol
  SYSTEM     Dhan token (expiry, auto-renewal, renew / paste) · health of every service · the ranker's own
             scorecard · walk-forward backtest · data-lake coverage
```

It runs out of the box on a built-in **simulator** that emits byte-identical Dhan packets, so you can
use and test everything before you pay for an API plan.

---

## Which Dhan API plan to buy

| | **Data APIs — ₹499/month + GST** | **Pro Traders — ₹999/month + GST** |
|---|---|---|
| Live market feed (ticker / quote / full packets, 5 sockets × 5,000 instruments) | ✓ | ✓ |
| 20-level market depth | ✓ | ✓ |
| Option chain + expiry list | ✓ | ✓ |
| Historical daily + intraday bars (the 5-year backfill) | ✓ | ✓ |
| **News headlines API** (the news-catalyst factor, NEWS desk) | — | ✓ |
| **CAS feed** (closing auction session after 15:30, recorded for next-day context) | — | ✓ |
| Market movers, 200-level depth, technical APIs | — | ✓ |

**Recommendation: ₹999 Pro Traders.** News carries 14% of the intraday score, and catalysts are the
most common reason a stock runs 3–4% in a day. The app also works on the ₹499 plan: set `DHAN_PLAN=basic`
and the news/CAS parts switch off. The other factors still rank normally, because a factor with no data
z-scores to zero and drops out.

Prices are from Dhan's plan page — check the current price before buying. Start on the simulator; buy
the plan when you are ready to run the backfill.

---

## Quick start

### Option A: Docker (recommended)

```bash
cp .env.example .env            # defaults run the simulator
docker compose up -d --build
# open http://localhost:8000
```

### Option B: plain Python, no Docker

Requirements: Python 3.11+ and a Redis server (`docker run -d -p 6379:6379 redis:7-alpine`, or Memurai
on Windows). The built UI is included; Node 20+ is only needed if you change the frontend
(`python run.py --build-ui`).

```bash
pip install -r backend/requirements.txt
cp .env.example .env
python run.py --speed 20      # simulator at 20×: a full session in ~20 minutes
# open http://localhost:8000
```

`run.py` starts the services (ingestor, analytics, geo, gateway, and the backfill worker behind /admin → HISTORY),
prefixes their logs and stops them all on Ctrl+C. It rebuilds the UI by itself when the frontend sources are newer
than the last build (e.g. after a `git pull`) and npm is installed.
For frontend hot-reload during development use `./scripts/dev.sh` (Vite on :5173).

---

## Going live with Dhan

1. **Get API access**: on web.dhan.co, open your profile's DhanHQ API section, subscribe to a data plan
   and generate an access token. It lasts 24 hours; from then on TradeDesk renews it by itself (below).
2. **Configure** `.env`:
   ```
   DATA_SOURCE=dhan
   DHAN_CLIENT_ID=1000000001
   DHAN_ACCESS_TOKEN=eyJ...
   DHAN_PLAN=pro            # or basic for the ₹499 plan
   # optional, for recovery after downtime (see "Dhan token" below):
   # DHAN_PIN=1234
   # DHAN_TOTP_SECRET=JBSWY3DPEHPK3PXP
   ```
3. **Start the stack** (`docker compose up -d --build` or `python run.py --live`), ideally before 09:15 IST.
4. **History loads itself** (`AUTO_HISTORY=true`, the default). Dhan's live feed sends a previous close
   only for indices, so on start-up and every morning at 08:40 the ingestor pulls from Dhan's
   historical API, only what is missing:
   * previous close of every stock and index → % change, gaps, breadth, movers, ranking
   * yesterday's close and open interest of every near-month future → the F&O build-up board
   * ~1 year of daily bars (`AUTO_HISTORY_DAYS=400`) → ATR, beta, prior-day levels, every swing setup
   * 30 days of 1-minute bars → the time-of-day volume curve behind RVOL
   * today's 1-minute bars so far when you start after 09:15 → opening range and full charts

   The first run takes about 3 minutes and ~750 of the 7,000 daily calls; later days ~400. Progress is
   on the admin page's SYSTEM tab (**Auto history**). If you start after 09:30 that day's picks can't be locked
   on time, so the Intraday desk shows a live ranking marked *provisional* and the day isn't scored.
5. **Optional, for research**: the full 5-year archive for the walk-forward backtest — open
   **/admin → HISTORY** and press **Start pull** (below). The pull follows the daily token renewal; a pull stopped by
   a rejected token resumes as soon as a new one is in place.

**Test the live path without an account or market hours**: `scripts/mock_dhan.py` is a local stand-in
for Dhan (scrip master, both websockets, historical / intraday / option-chain REST) with Dhan's quirks:
security ids that overlap across segments, previous close only for indices, zero IV on deep in-the-money
strikes. The run commands are in the script's docstring.

Rate limits are enforced in code: 5 data calls/s, 7,000/day (a budget shared through Redis between
the live system and the backfill), quotes 1/s, option chains 1 per 3 s per underlying and expiry.
Dhan counts the per-second limits per account, so they are shared too: the ingestor, the gateway, the backfill
worker and the CLI reserve their turns in Redis (`td:rl:*`, one queue, at most 5 calls in any second), and a
429 pauses all of them briefly; without Redis each process paces itself.

---

## Dhan token: daily expiry and automatic renewal

Access tokens from Dhan Web expire 24 hours after they are generated. TradeDesk keeps its token alive, so a
running deployment never needs a daily paste:

* **Renewed every day.** While the token is valid the ingestor swaps it for a fresh 24-hour one with Dhan's
  RenewToken call: in the 45-minute window from `DHAN_TOKEN_RENEW_AT` (08:30 IST, before the market opens) or,
  at the latest, 2 hours before it expires, weekends included. A renewal lasts until the same time the next day,
  so renewals land about once a day, outside market hours. Each new token is checked with Dhan (`/profile`,
  same client id, no data-API quota) before it is used; a refused one leaves the old token in place.
* **Every service follows.** The current token lives in Redis and in `data/auth/dhan_token.json` (file mode 0600,
  because Docker's Redis keeps nothing across restarts). The gateway, the backfill worker, `backfill/backfill.py`
  and the ingestor's REST calls pick up a new token within 30 s, and the websockets reconnect with it within
  seconds. On start-up the token with the latest expiry wins, so an old `DHAN_ACCESS_TOKEN` left in the
  environment no longer hides a newer one.
* **After downtime.** Dhan renews only a token that is still valid. If the server was off long enough for it to
  lapse, paste a new one on **/admin → SYSTEM → DHAN TOKEN** (no restart needed), or let the app log in by itself:
  enable TOTP on Dhan Web (DhanHQ Trading APIs → Setup TOTP), then put your Dhan PIN in `DHAN_PIN` and the text
  under the QR code in `DHAN_TOTP_SECRET`. **These two give full access to your Dhan account: keep them only in
  `.env` on the server, never in git or anywhere else.** If Dhan refuses them twice in a row, automatic login
  pauses (repeated wrong PINs can lock the account) until you fix them and restart the ingestor or press
  **Renew now**.
* **The DHAN TOKEN panel** at the top of /admin → SYSTEM (`F3`) shows the status (valid, expiring soon, expired,
  missing), the expiry with a countdown, where the token came from (`.env`, auto-renewed, TOTP, pasted), the next
  renewal, the data plan from Dhan's last check and the last error, with **Renew now** and **Paste new token**.
  The token itself is never shown, logged or returned by the API (`GET /api/admin/dhan-token`, admins only).

| `.env` | default | |
|---|---|---|
| `DHAN_TOKEN_AUTO_RENEW` | `true` | `false`: no renewals, replace the token by hand every day |
| `DHAN_TOKEN_RENEW_AT` | `08:30` | start of the daily renewal window (IST) |
| `DHAN_PIN`, `DHAN_TOTP_SECRET` | empty | both set: a new token is generated by itself once the old one has lapsed |

---

## Logins and the admin page

TradeDesk has user accounts with two roles. **Viewers** use the terminal (`/`). **Admins** also open the
separate admin page (`/admin`): USERS, HISTORY and SYSTEM.

* **First admin.** Put `ADMIN_USER` and `ADMIN_PASSWORD` in `.env` and restart (needed for Docker and remote
  servers). Or, on the host itself, open `http://localhost:8000/login` while no accounts exist: it offers a
  one-time **Create the admin account** screen.
* **Once any account exists, everyone logs in** at `/login`. Sessions last `SESSION_DAYS` (30). They're an HttpOnly
  cookie signed with `data/auth/secret.key`, so they survive restarts. Passwords are scrypt-hashed in
  `data/auth/users.json`.
* **USERS tab**: add users (viewer or admin), change roles, set a new password, disable, or delete them. Disabling a user or
  changing their password signs them out everywhere. The last admin can't be removed or demoted, and nobody
  can disable or delete their own account.
* **Access keys still work.** `ACCESS_KEYS` share links (`?key=…`) open the terminal for friends without
  accounts, and `/login` has an "I have an access key" option. They never open `/admin`.
* With no accounts and no access keys, the terminal is open as before (local use), but `/admin` always needs
  an admin login.
* **Behind a reverse proxy or tunnel** (nginx, Caddy, Cloudflare Tunnel): forward `Host` (or set
  `X-Forwarded-Host`), `X-Forwarded-For` and `X-Forwarded-Proto`, and use `ADMIN_USER`/`ADMIN_PASSWORD` for the first
  admin. The setup screen is switched off for requests that come through a proxy.

## Research desks: AI analyst, Screener, Monte Carlo, Quant

Four desks for researching any NSE stock (type a name or symbol in the box at the top of each; all four follow
the same stock). They work from the daily bars in the lake — the 5-year pull from **/admin → HISTORY** makes them
much better — plus Screener.in pages and, for the written analysis, an AI model. A stock outside the lake is fetched
from Dhan in one call (10 years of daily bars) when the token is valid, at most `RESEARCH_FETCH_PER_DAY` (300) times a
day, and only while more calls are left than the live desks need for the rest of the day (2 000 on Pro, 800 on
Basic, or `RESEARCH_FETCH_RESERVE` if higher).

**Nothing here predicts prices with certainty, and the desks say so.** Every forecast is a probability range, and
every model shows how good it has actually been on data it never saw.

* **F8 AI ANALYST** — press *Analyse*. It assembles a dossier — price & technicals (returns, CAGR, volatility,
  drawdowns, moving averages, RSI, 52-week range, support/resistance, beta and relative strength vs NIFTY),
  fundamentals from Screener.in, the quant model, a Monte Carlo run with its calibration check, market context
  (NIFTY trend, VIX, breadth, sectors) and recent news — and asks the AI model for a view for **6 months, 1, 2 and
  3 years**: stance, confidence, bull / base / bear targets with probabilities, drivers, risks, catalysts and what
  would change the view. Each base target is checked against the Monte Carlo 5–95 % band. The dossier the model saw
  is shown underneath, so every claim can be traced to a number.
  Configure the model in `.env`: `AI_PROVIDER=anthropic` (default, Claude via the official SDK, `AI_MODEL` defaults
  to `claude-opus-5-5`) or `AI_PROVIDER=openai` with `AI_BASE_URL` for any OpenAI-compatible API (OpenAI,
  OpenRouter, Groq, Together, DeepSeek, Gemini's OpenAI endpoint, a local Ollama). Without a key you still get the
  full dossier and a rule-based view. Reports are cached per stock per day; `AI_DAILY_LIMIT` and
  `AI_USER_DAILY_LIMIT` cap spend.
* **F9 SCREENER** — the company's Screener.in page as data: top ratios, pros & cons, compounded growth, quarterly and
  annual P&L, balance sheet, cash flow, ratios, shareholding and peers, plus derived metrics (CAGRs, margins,
  ROE/ROCE, debt/equity, interest cover, free cash flow, cash conversion, PEG, promoter/FII/DII changes) and a
  20-check scorecard (quality, growth, valuation, health, ownership) whose rules are printed next to each check.
  Banks and NBFCs are judged on lender metrics. Pages are cached `SCREENER_TTL_HOURS` and fetched one at a time
  (`SCREENER_MIN_INTERVAL` s apart); search URLs are never used. Screener's terms of use apply — personal research.
* **F10 MONTE CARLO** — simulate 2 000–25 000 paths up to 3 years. Models: GARCH(1,1) with fat-tailed shocks
  (volatility clusters and mean-reverts), block bootstrap of the stock's own returns, Merton jump-diffusion,
  a 2-state regime-switching model, plain GBM, or an ensemble of the first four (default). Drift: CAPM
  (risk-free + β × equity risk premium, default), shrunk historical mean, zero, or the quant model's 1-month view
  (only when that model has a measured edge). Shows the fan chart, the distribution of the final price, the
  6M/1Y/2Y/3Y percentiles, VaR/CVaR, drawdowns, the odds of touching your target prices, and every input it used.
  **Calibration**: *Backtest this model* re-fits the model at up to 80 past dates using only data before each one
  and counts how often the real price landed inside the 50/80/90 % bands — e.g. RELIANCE's 1-month 90 % bands held
  92.5 % of the time over 40 past dates. It also reports the CRPS skill against a plain random walk and the
  number of independent windows behind the verdict.
* **Alt+Q QUANT** — a pooled logistic model across every stock in the lake, on 26 causal features (Kalman trend,
  momentum, mean reversion, GARCH volatility, HMM regime, Hurst exponent, RSI, 52-week position, residual momentum
  vs NIFTY…), tested walk-forward (expanding window, purge + embargo, calibrated on earlier folds only). It gives
  P(up) for the next day, week and month, expected return, 68/95 % price ranges, a 21-day forecast cone, what is
  driving the model, a leaderboard, and the model's out-of-sample AUC / hit rate / Brier per horizon. **A horizon
  without a measured edge is labelled "no reliable edge" and its signal is held at NEUTRAL** — on the current
  5-year lake that is the honest result for 1 day and 1 month (AUC ≈ 0.52), with only a weak edge at 1 week.
  It retrains daily in the background (~1 minute); admins can retrain on demand.

## Admin page → HISTORY: pull the 5-year history

Open **/admin → HISTORY**, choose what to pull (daily and/or 1-minute bars, 1–5 years, all F&O symbols or a list),
check the preview — how many requests are still needed, how many calls are left today, how many days of
quota it spans — and press **Start pull**. It runs wherever TradeDesk runs (your PC, a VPS, Docker), writes
to that machine's lake, and the desk shows progress, the Dhan calls used today, every symbol's coverage and
an activity log.

The pull is done by the **backfill worker** (`python -m td.services.backfiller`, started by `run.py` and by
`docker compose up`), a separate process, so it cannot slow down or crash the live desks:

* **Paced**: 3 requests/s by default (Dhan allows 5/s in total), slowed to 1/s or paused in market hours
  (09:00–15:40 IST, your choice), and it waits while the ingestor loads its own history (start-up and 08:40).
* **Quota-safe**: it shares the 7,000-calls/day budget with the live app and keeps a reserve (2,500 by
  default; at least 2,000, which is what the live app itself uses on a trading day) for it. When the rest is used it waits and **resumes by itself after midnight IST**; 5 years of
  1-minute bars for the whole F&O list is ~4,600 requests, so about a day and a half.
* **Resumable**: what has been downloaded is remembered as date ranges (`data/backfill/state.json`), the job in
  `data/backfill/job.json`. A restart, a redeploy or Stop + Start carries on where it left off; ranges already
  in the lake are never fetched again.
* **One at a time**: a Redis lock keeps the desk and `backfill/backfill.py` from pulling together.

Only admin accounts can open it (see above).

## The 5-year backfill CLI (same engine, shared lake)

`backfill/backfill.py` is the same engine for scripts and cron, deliberately separate from the live system. It writes plain Parquet files
(`data/lake/dhan/<interval>/<SYMBOL>.parquet`, columns `ts open high low close volume oi`) that the live
analytics, the research scripts and anything else you build can read, with pandas, DuckDB or Polars.

```bash
python backfill/backfill.py plan    --intervals D,1 --years 5        # how many calls / days of quota
python backfill/backfill.py run     --intervals D,1 --years 5        # resumable across days (data/backfill/state.json)
python backfill/backfill.py update                                   # incremental, run daily after close
python backfill/backfill.py options --underlyings NIFTY,BANKNIFTY --strikes 2 --years 1   # rolling ATM±N options
python backfill/backfill.py status                                   # coverage per interval
python backfill/backfill.py sim                                      # synthetic lake for the simulator
```

It uses the token the ingestor keeps renewed (Redis, else `data/auth/dhan_token.json`), not the one in `.env`,
so a daily cron run keeps working after the first day.
5- and 15-minute bars are resampled from 1-minute bars on the fly, so `D,1` is all you need.
Daily bars take one request per symbol (Dhan returns 10+ years at once); 1-minute bars 90 days per request.
`--rate` (default 3/s) and `--reserve` (default 800 calls) keep room for the live system.
Docker: `docker compose run --rm tools python /app/backfill/backfill.py run --intervals D,1 --years 5`.

---

## Architecture

```
                 ┌──────────────── one machine (your PC or a small VPS) ──────────────┐
  Dhan WS feed ─▶│ INGESTOR  the only Dhan client: market feed (sharded sockets),     │
  Dhan REST    ─▶│           20-depth, CAS, option chains, news · token buckets ·     │
                 │           daily budget · tick recorder (Parquet)                   │
                 │        │ XADD td:ticks / td:news / td:events (Redis Streams)        │
                 │        ▼                                                            │
                 │ ANALYTICS  per-instrument state, minute bars, factor ranker,        │
                 │            09:30 lock + scorecard, option analytics, swing scanner, │
                 │            alerts → SET td:snap:<topic> + PUBLISH td:pub:<topic>    │
  free world   ─▶│ GEO        AIS, GDELT, FIRMS, ADS-B, PortWatch, GPR, FRED →         │
  feeds          │            chokepoints, fires at sites, airspace, geo-risk,         │
                 │            market impact → td:snap:geo:* (pauses unwatched feeds)   │
                 │        │ Redis pub/sub                                              │
                 │        ▼                                                            │
                 │ GATEWAY ×N  stateless FastAPI: REST + WebSocket, per-client topic   │
                 │            subscriptions, newest-frame coalescing, access keys,     │
                 │            serves the React terminal                                │
                 └────────┬────────────────────────────────────────────────────────────┘
                          ▼
        browsers: you · friends · phone   (each one = 1 websocket, 0 Dhan calls)
```

* **One Dhan connection, many viewers.** Only the ingestor touches Dhan, so ten viewers use exactly the
  same API quota as one.
* **Snapshots, not tick firehoses.** Analytics publishes compact snapshots (rank, top picks, quotes,
  chains…) once a second. A gateway keeps only the newest frame per topic for each client, so a slow
  phone on 3G never backs up the server.
* **Restart-safe.** Snapshots, minute bars, the 09:30 lock and alerts live in Redis. Restart the
  analytics or a gateway mid-session and it carries on where it left off.
* **Horizontal.** Gateways are stateless. If you ever need more, run several behind a reverse proxy
  (nginx, Caddy, Traefik).
* **Same code live and simulated.** Market time comes from the batch clock, so the simulator at 20× runs
  exactly the live code. The backtest calls the same ranker functions.

---

## Sharing with friends (no extra cost)

1. Give each person a key in `.env`: `ACCESS_KEYS=me:k8s7f2,ravi:p3x9q1,neha:z7m2c4` and restart.
2. Make the gateway reachable:
   * **Tailscale** (recommended): free and private. Install it on your machine and theirs, then share
     `http://<your-tailscale-name>:8000/?key=p3x9q1`.
   * **Cloudflare Tunnel**: gives you an https URL without opening router ports.
   * A small VPS running `docker compose up -d`.
3. Friends open the link once. The key is remembered in their browser, and a wrong key gets an
   access-key prompt.

Prefer user accounts (admin page → USERS) over access keys: each person gets their own login, and you can
sign them out or remove them at any time. Never expose the gateway to the internet without accounts or
`ACCESS_KEYS`, and put it behind https (Tailscale, Cloudflare Tunnel, a reverse proxy) so passwords and session
cookies are encrypted.

> **Check the terms first.** Dhan's API data is licensed for your own use, and NSE regulates
> redistribution of real-time market data. Sharing a live feed with other people can count as
> redistribution. Keep it to a small private group, don't charge for it, and read Dhan's API terms
> before you share. The simulator has no such restriction.

---

## GEO desk (F7): world events that move Indian markets

A dark world map centred on India's trade routes. On the left there's one on/off switch per view. Switching a view
off removes its map layer and panel and unsubscribes your browser from that feed. When nobody is watching ships or
aircraft, the GEO worker also pauses those feeds. The geo-risk number also sits in the top bar on every desk.

| View | What you see | Source (free) | Key |
|---|---|---|---|
| Ship movement (live) | tankers, cargo, passenger, military vessels with heading; Gulf, Red Sea, Arabian Sea, Bay of Bengal, Malacca | AISStream.io websocket (terrestrial AIS) | free `AISSTREAM_API_KEY` |
| Chokepoints & ports | Hormuz, Bab el-Mandeb, Suez, Malacca, Cape, Bosporus… with status NORMAL / WATCH / STRAINED / DISRUPTED; Indian and Gulf ports | IMF PortWatch daily transits (7-day vs usual) + live AIS + nearby conflict | none |
| War & conflict events | air strikes, fighting, bombings, blockades, mobilisation; severity-sized dots, source links | GDELT 2.0 event files, every 15 min | none |
| Geopolitical risk index | live India-weighted geo-risk 0–100 with its five components + the academic GPR index | own model + Caldara–Iacoviello GPR (daily, and India monthly) | none |
| Fires / explosions / refinery hits | satellite hotspots; refineries, terminals and gas fields flagged ANOMALY when far above their own usual flaring | NASA FIRMS (VIIRS S-NPP, NOAA-20, NOAA-21) | free `FIRMS_MAP_KEY` |
| Satellite imagery | daily true-colour (VIIRS, MODIS), night lights, Sentinel-2 10 m mosaic; date picker; links to NASA Worldview and Copernicus Browser per site | NASA GIBS, EOX (loaded directly by the browser) | none |
| Military / air traffic | military aircraft worldwide + an airspace watch (Iran, Iraq, Gulf, Israel, Pakistan, Delhi, Mumbai, Red Sea, Ukraine): traffic vs its usual level for the hour — CLOSED? / REDUCED | adsb.lol (or airplanes.live) | none |
| Oil rigs & supply | refineries, terminals, offshore fields, LNG on the map; MCX crude & gas live; Brent, WTI, Henry Hub; Hormuz tanker flow | Dhan feed (MCX) + FRED | none |
| India macro | USD/INR futures and spot, MCX gold & silver, India VIX, US 10Y, dollar index; RBI / FOMC / CPI / GDP calendar | Dhan feed (NSE currency, MCX) + FRED | none |
| Market impact | each signal mapped to the NSE names it tends to move (▲/▼) with the reason; click a name to open it | rules in `backend/td/geo/engine.py` | — |

How the geo-risk index works (0–100): conflict 35% (India-weighted event severity in the last 24 h, compared with
its own trailing median once a day of history exists), shipping 25% (chokepoint status), energy 15% (fire
anomalies at energy sites), airspace 10% (watched airspaces far below normal), markets 15% (crude and rupee moves,
India VIX). It reads LOW < 25 ≤ ELEVATED < 45 ≤ HIGH < 65 ≤ SEVERE. Treat it as a heat gauge for your own desk,
not a forecast.

Things to know:
* On the simulator (`DATA_SOURCE=sim`), every GEO view runs on simulated world feeds marked SIM: ships on real
  lanes, conflict in the usual hot spots, a refinery incident, an airspace closure. Set `GEO_SOURCE=live` to use real
  feeds while the market side stays simulated.
* Coverage gaps are real. Free AIS comes from shore receivers, so open ocean and some Gulf waters are thin. ADS-B
  misses aircraft with transponders off. GDELT geolocation is automated and noisy, so it can misplace or duplicate
  an event. PortWatch transits lag by a few days.
* Refineries flare every day. A hotspot isn't a hit; ANOMALY means far above the site's own usual. Open the
  imagery link before acting.
* Site coordinates are approximate. Add or correct sites in `data/config/geo_sites.json` (same fields as
  `backend/td/geo/sites.py`), and add calendar dates in `data/config/macro_calendar.json`.
* `GEO_TO_NEWS=true` sends severity-3 signals to the NEWS desk, where they feed the ranker's catalyst factor.
  It's off by default.
* A view showing ERROR or WAITING: hover its badge for the reason, then check every feed from the server itself with
  `docker compose exec geo python -m td.geo.diagnose` (or `cd backend && python -m td.geo.diagnose`). It prints
  one OK/FAIL line per source and never prints your keys. Feeds that work at home can be refused from cloud server
  addresses. FRED is the usual one, and a free `FRED_API_KEY` fixes it. When adsb.lol rate-limits the server, the
  air view keeps the aircraft it has and pauses for 30 s to 5 min instead of erroring.
* Military aircraft load when someone opens the air view (the first batch within a few seconds, then every 2 min).
  A ship's type (tanker, cargo…) arrives in a separate AIS report every 6 minutes, and shore receivers miss many
  of them. So on a fresh install most ships start grey ("unknown") and get labelled over the first hours. The
  worker keeps learning types even while nobody views ships (type reports only, a few a minute). Learnt types are
  remembered in Redis for 14 days, so restarts keep them.

## How the intraday picks work

At every refresh, all ~180 F&O stocks are scored on eight factors. Each factor is robust z-scored
across the universe (median/MAD, clipped at ±3), so the score says *which stocks are most in play right
now relative to the others*. It is never an absolute forecast.

| Factor | Weight | What it measures |
|---|---|---|
| Relative volume | 22% | volume vs the same minute over the last 14 sessions (log) |
| Relative strength | 18% | return since previous close minus β × NIFTY |
| News catalyst | 14% | time-decayed, novelty-weighted headline score (event type × direction) |
| VWAP position | 14% | distance from VWAP in ATR units |
| Opening range | 12% | position in / breakout of the first 15 minutes |
| Futures OI | 10% | long build-up > short covering > long unwinding > short build-up |
| vs sector | 6% | stock vs its sector median, so one sector move is not counted five times |
| Book imbalance | 4% | resting bid/ask imbalance (spoofable, so a small weight) |

* Before 09:30 the list is **provisional**. At `LOCK_TIME` it **locks** with entry, an invalidation level
  (opening-range low/high or 0.6 ATR) and a 1-ATR target, then it is tracked to the close.
* Filters: liquidity (₹5 Cr traded by the lock), RVOL ≥ 1.2, direction must agree with VWAP and
  relative strength, and at most two picks per sector. If fewer than five stocks qualify, it
  **abstains** rather than forcing a pick.
* At 15:30 the **scorecard** records the rank IC, how many picks finished in the day's top-5 and top-20,
  the green %, and excess return vs the universe. This is the system's own track record, on /admin → SYSTEM.

### Measure it before you trust it

```bash
python research/evaluate_intraday.py --lake data/lake/dhan --days 250 --cost-bps 8
python research/evaluate_intraday.py --fit --write-weights   # IC-weighted weights, 60/40 train/test split
```

The script replays every day in the lake through the **same ranker code**, using only data available
before each day, and reports daily IC, the IC of each factor on its own, top-5/top-20 hits vs random,
and pick returns net of costs. `--write-weights` saves new weights only if they beat the current ones
out of sample. The analytics worker loads them from `data/config/ranker_weights.json`.

What good looks like: a mean rank IC of **0.03–0.08** that holds across months is a real edge. Picks
landing in the day's top-20 at 2–3× the random rate is good. Nothing will name the exact top-5
gainers every morning. Treat the picks as a ranked shortlist.

History contains only prices and volume, so the backtest scores OI, news and book imbalance as zero.
The live scorecard covers all eight factors.
On the **synthetic** lake the backtest is only a plumbing test: its minute bars are a Brownian bridge to
each day's close, so early moves mean-revert by construction.

---

## Swing, F&O and SMC

* **Swing setups** (daily bars plus today's live bar): breakout (20-day high on ≥1.5× volume), VCP
  (10-day ATR < 0.75× 50-day ATR near highs), pullback to EMA20 in a full uptrend, momentum (RSI > 60 in
  trend), breakdown (short). Each setup shows **receipts**: how often it hit +4% before −2% within 10 days
  across the whole lake. A hit rate near the base rate means no edge.
* **F&O**: futures OI build-up quadrants. Option chain with Black-Scholes IV and greeks, PCR (OI and
  volume), max pain, call/put walls, net gamma exposure by strike, gamma-flip level, 25-delta skew and
  the straddle-implied expected move.
* **Smart Money Concepts** on every chart, **non-repainting**: swing points are confirmed only after
  `k` bars, BOS/CHoCH are marked on the bar that broke structure, fair-value gaps are tracked until
  filled, order blocks are the last opposite-colour candle before the impulse that broke structure
  (dropped once price returns to them), and liquidity sweeps are wicks through a confirmed swing that
  close back inside.

---

## Keyboard

| Key | Action |
|---|---|
| `/` or `Ctrl+K` | command line |
| `F1`…`F10` / `Alt+1`…`0` | switch desk |
| `Alt+Q` | QUANT desk |
| `RELIANCE` ⏎ | load a symbol into chart and depth (on a research desk: into that desk) |
| `TOP5` `MKT` `FNO` `SWING` `NEWS` `ALRT` `GEO` `AI` `SCRN` `MC` `QUANT` | open a desk |
| `CHAIN BANKNIFTY` | option chain for an underlying |
| `HELP` | shortcuts |
| `F1` `F2` `F3` / `Alt+1`…`3` on /admin | USERS · HISTORY · SYSTEM (Dhan token, services, scorecard) |

---

## Project layout

```
backend/td/
  config.py            settings (.env)
  dhan/                REST client, websocket feeds (sharded), binary packet parser, instruments,
                       token keeper (daily access-token renewal)
  sim/simulator.py     market simulator (Dhan-identical packets) + synthetic history
  services/            ingestor.py · analytics.py · geo.py · gateway.py · recorder.py · bootstrap.py (auto history)
  geo/                 world-event sources (GDELT, FIRMS, AIS, ADS-B, PortWatch, GPR, FRED), simulator, risk/impact engine, sites
  analytics/           intraday ranker · option math · SMC · swing · news · market breadth · alerts
  research/            research desks: tsmath (GARCH, Kalman, HMM, bootstrap…) · montecarlo · quant ·
                       screener (Screener.in) · llm (Claude / OpenAI-compatible) · ai (the analyst) · data
  lake.py              Parquet lake (+ DuckDB coverage)
  ratelimit.py         token buckets, per-second limits shared across processes (Redis), daily budget, option-chain window
backend/tests/         pytest (packet layouts, BS/IV, SMC non-repainting, ranker, news, swing, lake, geo parsers & rules, live-mode regressions,
                       Dhan token keeper (renewal, TOTP, followers, admin API),
                       research maths, Monte Carlo calibration, quant causality, Screener parser, AI client & analyst, research API)
backfill/backfill.py   5-year history tool
research/              evaluate_intraday.py (walk-forward backtest + weight fitting)
scripts/               mock_dhan.py (local fake Dhan for testing the live path) · dev.sh
frontend/              React 19 + TypeScript + lightweight-charts terminal; MapLibre GL map for the GEO desk
run.py                 one-command launcher · docker-compose.yml · Dockerfile · .env.example
data/                  lake/ recorder/ scorecards/ research/ config/   (created at runtime)
```

Tests: `cd backend && python -m pytest -q`

---

## Troubleshooting

* **Blank desks / "Waiting…"**: check /admin → SYSTEM. INGESTOR and ANALYTICS should show UP.
  `docker compose logs -f ingestor analytics`.
* **Dhan errors 806–810**: 806 data plan not active, 807 token expired, 808/809 authentication or invalid
  token, 810 wrong client ID. Open /admin → SYSTEM → DHAN TOKEN: **Renew now** while the token is still valid,
  otherwise **Paste new token** (or set `DHAN_PIN` + `DHAN_TOTP_SECRET`, see "Dhan token" above).
* **"Daily data-API budget exhausted"**: the backfill and the live system share the 7,000/day budget.
  Run large backfills with `--reserve 1500` on trading days, or on weekends.
* **Ranking empty, breadth covers 2 stocks, NIFTY/BANKNIFTY missing from the top bar, F&O build-up
  all zeros, swing empty** (live mode): you're on a build older than the auto-history fix. Rebuild:
  `docker compose up -d --build`, then watch **Auto history** on /admin → SYSTEM reach READY (~3 min).
* **Option chain shows IV 0 / Δ 0 on deep in-the-money strikes**: Dhan returns 0 for strikes that barely
  trade; TradeDesk solves the IV from the last price. When that price is below intrinsic (stale) IV shows
  "—" and Δ shows ±1.
* **RVOL looks wrong on day one**: the 1-minute history is missing, so a generic U-shaped volume curve
  is used. Auto history fills it within a few minutes; check /admin → SYSTEM.
* **Redis memory**: the tick stream is capped by `STREAM_MAXLEN` (default 1,500 × 100 ms batches). History
  goes to `data/recorder` on disk.

---

This is a research and monitoring tool. It does not give investment advice and it does not
place orders. Past scorecards do not guarantee future results.
