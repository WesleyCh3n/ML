import numpy as np
import matplotlib.pyplot as plt
import time as t

class das():
    def __init__(self):
        self.obj = 12 
    def __str__(self):
        return "Testing"

if __name__ == "__main__":
    ts = t.time()
    a = das()
    print(a)
    te = t.time()
    print(f"te - ts = {te-ts:.2f}")
    print(a.obj)
