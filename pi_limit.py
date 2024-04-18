import argparse
import os

import torch
import os
import logging
import numpy as np
import pandas as pd


from RollingWindow import RollingWindow

from TorchRunner import TorchRunner


from MSRR import MSRR
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)


def set_seed(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.info(f"{'='*12}>\tSeed:\t{seed}")





if __name__ == "__main__":

    # Default value 1234
    set_seed(1234)
    data = pd.read_pickle("usa_131_ranked_large_mega.pickle")
    data["month"] = (
        pd.to_datetime(data.date).dt.year * 100 + pd.to_datetime(data.date).dt.month
    )

    window = RollingWindow(data=data, window_size=5) # years

    model = #TODO build a model here, 
    criterion = MSRR()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    runner = TorchRunner(
        True,
        5,
        2023,
        "./",
        epochs=10,
        data=data,
        resume=False,
        no_incremental=False,
        model_name="pi_limit",
    )

    runner.run(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        window=window,
    )
