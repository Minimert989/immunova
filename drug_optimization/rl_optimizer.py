# RL optimizer
import random

def optimize_combo():
    combos = [(a, b) for a in range(3) for b in range(3)]
    return random.choice(combos)