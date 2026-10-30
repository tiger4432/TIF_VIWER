"""Cache chip crops (factor 2) of a wafer as .npy: python cache.py OUTDIR NAME [NAME ...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
from tools.synth.chips import load_chips

for name in sys.argv[2:]:
    p = os.path.join(sys.argv[1], name + "_chips2.npy")
    if not os.path.exists(p):
        np.save(p, load_chips(sys.argv[1], name, 2))
        print("cached", p)
