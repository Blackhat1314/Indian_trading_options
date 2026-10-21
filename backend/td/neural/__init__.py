"""NeuralEdge — a neural network over price history, technicals, point-in-time fundamentals and news.

Trained on a PC with a GPU (`research/neural_train.py`), exported to ONNX, and run on the server with
onnxruntime (`td.neural.runtime`). The training data and the walk-forward rules are the QuantEdge ones
(td.research.quant), so the two models are scored on the same test periods and can be compared honestly.

Modules: fundamentals (Screener tables → what was public on each day), news (recorder archive → daily
features), dataset (the panel the network learns from), model (PyTorch, imported only for training),
train (walk-forward + export), runtime (server-side inference).
"""
