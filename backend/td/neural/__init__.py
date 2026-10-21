"""NeuralEdge — neural networks over price history, technicals, point-in-time fundamentals, news and 1-minute bars.

Trained on a PC with a GPU (`research/neural_train.py`), exported to ONNX, and run on the server with
onnxruntime. Two models:
  daily   P(up) / P(beats the median stock) after 1 day, 1 week, 1 month (served by `runtime` to the NEURAL desk)
  09:30   P(up by the close) / P(beats the day's median) at the TOP5 lock (run by the analytics worker via `live`)
The training data and walk-forward rules are QuantEdge's (td.research.quant), so the models are scored on the same
test periods as their baselines (QuantEdge, gradient boosting, and for 09:30 the live TOP5 ranker).

Modules: fundamentals (Screener tables → what was public on each day), news (recorder archive → daily features),
minute (1-minute bars → one row per stock-day), dataset (the daily panel), intraday (the 09:30 panel and its
per-day features, shared with live scoring), model (PyTorch, imported only for training), train / train_intraday
(walk-forward, settings search, export), runtime (daily serving), live (09:30 serving).
"""
