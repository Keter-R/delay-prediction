import torch
import pandas as pd


def toLabels(y, max_delay, limit=30):
    # y: (batch_size, 1)
    # max_delay: float
    # return: (batch_size, 0/1)
    y = y * max_delay
    # check all values in y
    # if y > limit, set y to 1
    # else, set y to 0
    y = (y > limit).float()
    return y
