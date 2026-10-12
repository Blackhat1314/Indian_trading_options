"""Dhan v2 enums (from the official Annexure, docs.dhanhq.co/api/v2/guides/annexure)."""

SEGMENT_CODE = {
    "IDX_I": 0,
    "NSE_EQ": 1,
    "NSE_FNO": 2,
    "NSE_CURRENCY": 3,
    "BSE_EQ": 4,
    "MCX_COMM": 5,
    "BSE_CURRENCY": 7,
    "BSE_FNO": 8,
}
SEGMENT_NAME = {v: k for k, v in SEGMENT_CODE.items()}

# Feed request codes
REQ_CONNECT = 11
REQ_DISCONNECT = 12
REQ_SUB_TICKER = 15
REQ_UNSUB_TICKER = 16
REQ_SUB_QUOTE = 17
REQ_UNSUB_QUOTE = 18
REQ_SUB_FULL = 21
REQ_UNSUB_FULL = 22
REQ_SUB_DEPTH = 23          # full market depth (20/200) and CAS feed
REQ_UNSUB_DEPTH = 25
REQ_UNSUB_CAS = 24

# Feed response codes
RESP_INDEX = 1
RESP_TICKER = 2
RESP_QUOTE = 4
RESP_OI = 5
RESP_PREV_CLOSE = 6
RESP_MARKET_STATUS = 7
RESP_FULL = 8
RESP_DISCONNECT = 50
RESP_DEPTH_BID = 41
RESP_DEPTH_ASK = 51

# Fixed packet sizes for the live market feed (bytes, header included)
PACKET_SIZE = {
    RESP_INDEX: 16,
    RESP_TICKER: 16,
    RESP_QUOTE: 50,
    RESP_OI: 12,
    RESP_PREV_CLOSE: 16,
    RESP_MARKET_STATUS: 8,
    RESP_FULL: 162,
    RESP_DISCONNECT: 10,
}

# Websocket limits
MAX_CONNECTIONS = 5
MAX_INSTRUMENTS_PER_CONNECTION = 5000
MAX_INSTRUMENTS_PER_MESSAGE = 100
DEPTH20_MAX_INSTRUMENTS = 50

# Well known index security ids (IDX_I)
INDEX_IDS = {
    "NIFTY": 13,
    "BANKNIFTY": 25,
    "FINNIFTY": 27,
    "MIDCPNIFTY": 442,
    "SENSEX": 51,
}

DATA_API_ERRORS = {
    800: "Internal Server Error",
    804: "Requested number of instruments exceeds limit",
    805: "Too many requests or connections",
    806: "Data APIs not subscribed",
    807: "Access token is expired",
    808: "Authentication failed",
    809: "Access token is invalid",
    810: "Client ID is invalid",
    811: "Invalid expiry date",
    812: "Invalid date format",
    813: "Invalid SecurityId",
    814: "Invalid request",
}
