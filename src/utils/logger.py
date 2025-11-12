import logging, sys

def get_logger(name="app"):
    lg = logging.getLogger(name)
    if lg.handlers: return lg
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    lg.addHandler(h); lg.setLevel(logging.INFO)
    return lg
