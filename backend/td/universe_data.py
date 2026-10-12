"""Static reference data for the F&O universe.

Sector tags drive the sector heatmap, sector-neutral ranking and relative strength.
Dhan's scrip master has no sector column, so we keep a curated map here (enriched at
runtime from the Company Info API when available). `ref_px` and `vol` are only used by
the simulator to produce realistic-looking prices — live mode uses real Dhan prices.
"""

# symbol, sector, ref price (₹, approx), daily volatility (%)
FNO_STOCKS: list[tuple[str, str, float, float]] = [
    # Banks
    ("HDFCBANK", "BANK", 1980, 1.2), ("ICICIBANK", "BANK", 1420, 1.3), ("SBIN", "BANK", 880, 1.6),
    ("KOTAKBANK", "BANK", 2150, 1.4), ("AXISBANK", "BANK", 1210, 1.6), ("INDUSINDBK", "BANK", 840, 2.4),
    ("BANKBARODA", "BANK", 265, 2.0), ("PNB", "BANK", 118, 2.2), ("CANBK", "BANK", 128, 2.2),
    ("FEDERALBNK", "BANK", 215, 1.8), ("IDFCFIRSTB", "BANK", 78, 2.1), ("AUBANK", "BANK", 760, 2.0),
    ("BANDHANBNK", "BANK", 182, 2.4), ("RBLBANK", "BANK", 265, 2.5), ("UNIONBANK", "BANK", 142, 2.2),
    ("INDIANB", "BANK", 640, 2.1), ("YESBANK", "BANK", 22, 2.6),
    # Financials
    ("BAJFINANCE", "FIN", 980, 1.9), ("BAJAJFINSV", "FIN", 2050, 1.6), ("CHOLAFIN", "FIN", 1620, 2.0),
    ("SHRIRAMFIN", "FIN", 690, 2.2), ("MUTHOOTFIN", "FIN", 3150, 1.9), ("MANAPPURAM", "FIN", 285, 2.4),
    ("LICHSGFIN", "FIN", 620, 2.0), ("PFC", "FIN", 425, 2.3), ("RECLTD", "FIN", 405, 2.3),
    ("IRFC", "FIN", 135, 2.4), ("HDFCLIFE", "FIN", 780, 1.5), ("SBILIFE", "FIN", 1860, 1.4),
    ("ICICIGI", "FIN", 1990, 1.5), ("ICICIPRULI", "FIN", 650, 1.8), ("LICI", "FIN", 930, 1.6),
    ("HDFCAMC", "FIN", 5600, 1.8), ("ABCAPITAL", "FIN", 285, 2.3), ("LTF", "FIN", 245, 2.2),
    ("JIOFIN", "FIN", 318, 2.0), ("CDSL", "FIN", 1720, 2.8), ("BSE", "FIN", 2650, 3.2),
    ("MCX", "FIN", 8900, 2.8), ("ANGELONE", "FIN", 2650, 2.9), ("CAMS", "FIN", 4100, 2.2),
    ("IEX", "FIN", 145, 2.6), ("POLICYBZR", "FIN", 1880, 2.4), ("PAYTM", "FIN", 1280, 2.8),
    # IT
    ("TCS", "IT", 3150, 1.3), ("INFY", "IT", 1520, 1.5), ("HCLTECH", "IT", 1480, 1.5),
    ("WIPRO", "IT", 248, 1.6), ("TECHM", "IT", 1460, 1.7), ("LTIM", "IT", 5600, 1.8),
    ("MPHASIS", "IT", 2780, 2.0), ("COFORGE", "IT", 1720, 2.3), ("PERSISTENT", "IT", 5800, 2.3),
    ("OFSS", "IT", 8900, 2.2), ("KPITTECH", "IT", 1250, 2.5), ("TATAELXSI", "IT", 5400, 2.1),
    ("LTTS", "IT", 4300, 1.9), ("CYIENT", "IT", 1180, 2.2), ("NAUKRI", "IT", 1380, 2.0),
    ("TATATECH", "IT", 690, 2.2),
    # Auto
    ("MARUTI", "AUTO", 15800, 1.4), ("TMPV", "AUTO", 410, 2.0), ("M&M", "AUTO", 3450, 1.6),
    ("BAJAJ-AUTO", "AUTO", 8800, 1.5), ("HEROMOTOCO", "AUTO", 5400, 1.6), ("EICHERMOT", "AUTO", 6900, 1.6),
    ("TVSMOTOR", "AUTO", 3450, 1.7), ("ASHOKLEY", "AUTO", 145, 2.0), ("BHARATFORG", "AUTO", 1320, 2.0),
    ("BOSCHLTD", "AUTO", 38500, 1.5), ("MOTHERSON", "AUTO", 108, 2.1), ("EXIDEIND", "AUTO", 395, 2.0),
    ("MRF", "AUTO", 152000, 1.3), ("APOLLOTYRE", "AUTO", 495, 1.9), ("BALKRISIND", "AUTO", 2420, 1.8),
    ("TIINDIA", "AUTO", 3050, 2.0), ("SONACOMS", "AUTO", 470, 2.2), ("UNOMINDA", "AUTO", 1260, 2.2),
    # Energy / oil & gas
    ("RELIANCE", "ENERGY", 1420, 1.3), ("ONGC", "ENERGY", 245, 1.8), ("IOC", "ENERGY", 148, 1.9),
    ("BPCL", "ENERGY", 335, 1.9), ("HINDPETRO", "ENERGY", 445, 2.1), ("GAIL", "ENERGY", 185, 1.8),
    ("OIL", "ENERGY", 445, 2.3), ("PETRONET", "ENERGY", 285, 1.7), ("IGL", "ENERGY", 205, 1.9),
    ("ADANIENT", "ENERGY", 2450, 2.6),
    # Power
    ("NTPC", "POWER", 345, 1.5), ("POWERGRID", "POWER", 295, 1.4), ("TATAPOWER", "POWER", 395, 2.0),
    ("NHPC", "POWER", 82, 2.0), ("JSWENERGY", "POWER", 520, 2.4), ("TORNTPOWER", "POWER", 1380, 2.0),
    ("CESC", "POWER", 168, 2.1), ("ADANIGREEN", "POWER", 1020, 2.9), ("ADANIENSOL", "POWER", 880, 2.6),
    ("SJVN", "POWER", 98, 2.3),
    # Metals & mining
    ("TATASTEEL", "METAL", 168, 1.9), ("JSWSTEEL", "METAL", 1080, 1.8), ("HINDALCO", "METAL", 720, 1.9),
    ("VEDL", "METAL", 465, 2.2), ("SAIL", "METAL", 132, 2.3), ("JINDALSTEL", "METAL", 1010, 2.1),
    ("NMDC", "METAL", 74, 2.1), ("NATIONALUM", "METAL", 215, 2.5), ("HINDZINC", "METAL", 470, 2.2),
    ("COALINDIA", "METAL", 385, 1.6), ("HINDCOPPER", "METAL", 295, 2.9),
    # Pharma & healthcare
    ("SUNPHARMA", "PHARMA", 1680, 1.3), ("DRREDDY", "PHARMA", 1260, 1.5), ("CIPLA", "PHARMA", 1540, 1.4),
    ("DIVISLAB", "PHARMA", 6400, 1.7), ("LUPIN", "PHARMA", 1980, 1.7), ("AUROPHARMA", "PHARMA", 1120, 1.8),
    ("ZYDUSLIFE", "PHARMA", 980, 1.7), ("ALKEM", "PHARMA", 5450, 1.6), ("TORNTPHARM", "PHARMA", 3500, 1.4),
    ("BIOCON", "PHARMA", 365, 2.1), ("GLENMARK", "PHARMA", 1950, 2.2), ("LAURUSLABS", "PHARMA", 880, 2.3),
    ("MANKIND", "PHARMA", 2450, 1.7), ("APOLLOHOSP", "PHARMA", 7600, 1.5), ("MAXHEALTH", "PHARMA", 1180, 1.9),
    ("FORTIS", "PHARMA", 920, 2.0), ("SYNGENE", "PHARMA", 660, 1.9),
    # FMCG
    ("HINDUNILVR", "FMCG", 2480, 1.1), ("ITC", "FMCG", 410, 1.1), ("NESTLEIND", "FMCG", 1220, 1.1),
    ("BRITANNIA", "FMCG", 5900, 1.2), ("DABUR", "FMCG", 510, 1.3), ("MARICO", "FMCG", 720, 1.3),
    ("GODREJCP", "FMCG", 1180, 1.4), ("COLPAL", "FMCG", 2250, 1.3), ("TATACONSUM", "FMCG", 1120, 1.5),
    ("VBL", "FMCG", 480, 2.0), ("UNITDSPR", "FMCG", 1420, 1.5),
    # Cement & construction & infra
    ("ULTRACEMCO", "INFRA", 12100, 1.3), ("SHREECEM", "INFRA", 29500, 1.4), ("AMBUJACEM", "INFRA", 565, 1.7),
    ("ACC", "INFRA", 1850, 1.5), ("DALBHARAT", "INFRA", 2150, 1.7), ("LT", "INFRA", 3850, 1.3),
    ("ADANIPORTS", "INFRA", 1460, 1.9), ("GMRAIRPORT", "INFRA", 92, 2.2), ("NBCC", "INFRA", 115, 2.7),
    ("RVNL", "INFRA", 345, 2.8), ("IRCTC", "INFRA", 720, 1.7), ("CONCOR", "INFRA", 560, 1.8),
    ("DELHIVERY", "INFRA", 450, 2.4), ("INDIGO", "INFRA", 5650, 1.8),
    # Capital goods & defence
    ("HAL", "DEFENCE", 4650, 2.2), ("BEL", "DEFENCE", 410, 2.0), ("BDL", "DEFENCE", 1620, 2.9),
    ("MAZDOCK", "DEFENCE", 2850, 3.0), ("SOLARINDS", "DEFENCE", 14200, 2.4), ("BHEL", "CAPGOODS", 255, 2.4),
    ("SIEMENS", "CAPGOODS", 3150, 2.0), ("ABB", "CAPGOODS", 5450, 2.0), ("CGPOWER", "CAPGOODS", 720, 2.3),
    ("CUMMINSIND", "CAPGOODS", 3850, 1.9), ("POLYCAB", "CAPGOODS", 7300, 2.1), ("KEI", "CAPGOODS", 4100, 2.3),
    ("HAVELLS", "CAPGOODS", 1540, 1.6), ("DIXON", "CAPGOODS", 15800, 2.6), ("VOLTAS", "CAPGOODS", 1380, 2.0),
    ("BLUESTARCO", "CAPGOODS", 1850, 2.2), ("SUPREMEIND", "CAPGOODS", 4300, 1.9), ("ASTRAL", "CAPGOODS", 1450, 2.0),
    ("KAYNES", "CAPGOODS", 6400, 3.0), ("PGEL", "CAPGOODS", 780, 3.0),
    # Consumer & retail
    ("TITAN", "CONSUMER", 3550, 1.5), ("ASIANPAINT", "CONSUMER", 2450, 1.4), ("BERGEPAINT", "CONSUMER", 545, 1.5),
    ("PIDILITIND", "CONSUMER", 1490, 1.3), ("TRENT", "CONSUMER", 4900, 2.3), ("DMART", "CONSUMER", 4300, 1.7),
    ("PAGEIND", "CONSUMER", 43500, 1.5), ("JUBLFOOD", "CONSUMER", 610, 1.9), ("KALYANKJIL", "CONSUMER", 520, 2.7),
    ("NYKAA", "CONSUMER", 235, 2.4), ("ETERNAL", "CONSUMER", 320, 2.6), ("SWIGGY", "CONSUMER", 420, 2.8),
    ("INDHOTEL", "CONSUMER", 760, 1.8), ("DLF", "REALTY", 780, 2.2), ("GODREJPROP", "REALTY", 2250, 2.5),
    ("OBEROIRLTY", "REALTY", 1650, 2.3), ("PRESTIGE", "REALTY", 1650, 2.6), ("LODHA", "REALTY", 1250, 2.5),
    ("PHOENIXLTD", "REALTY", 1620, 2.1),
    # Telecom & media
    ("BHARTIARTL", "TELECOM", 1950, 1.3), ("IDEA", "TELECOM", 8.5, 3.2), ("INDUSTOWER", "TELECOM", 385, 2.0),
    ("TATACOMM", "TELECOM", 1720, 1.9),
    # Chemicals & agri
    ("UPL", "CHEM", 680, 1.9), ("PIIND", "CHEM", 3700, 1.8), ("SRF", "CHEM", 2950, 1.7),
    ("TATACHEM", "CHEM", 920, 1.8), ("NAVINFLUOR", "CHEM", 4700, 2.0), ("COROMANDEL", "CHEM", 2250, 1.7),
    ("CHAMBLFERT", "CHEM", 520, 1.9),
]

SECTORS = ["BANK", "FIN", "IT", "AUTO", "ENERGY", "POWER", "METAL", "PHARMA", "FMCG", "INFRA",
           "DEFENCE", "CAPGOODS", "CONSUMER", "REALTY", "TELECOM", "CHEM"]

SECTOR_NAMES = {
    "BANK": "Banks", "FIN": "Financials", "IT": "IT", "AUTO": "Auto", "ENERGY": "Oil & Gas",
    "POWER": "Power", "METAL": "Metals", "PHARMA": "Pharma", "FMCG": "FMCG", "INFRA": "Infra & Cement",
    "DEFENCE": "Defence", "CAPGOODS": "Capital Goods", "CONSUMER": "Consumer", "REALTY": "Realty",
    "TELECOM": "Telecom", "CHEM": "Chemicals",
}

# index name -> (Dhan IDX_I security id, simulator reference level, daily vol %)
INDICES: dict[str, tuple[int, float, float]] = {
    "NIFTY": (13, 25450.0, 0.85),
    "BANKNIFTY": (25, 56800.0, 1.0),
    "FINNIFTY": (27, 26900.0, 1.0),
    "MIDCPNIFTY": (442, 13350.0, 1.2),
    "INDIAVIX": (21, 12.4, 4.0),
}

SECTOR_OF = {s: sec for s, sec, _, _ in FNO_STOCKS}
