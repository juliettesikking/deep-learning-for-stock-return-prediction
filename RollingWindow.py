import torch
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


class RollingWindow:
    """Keeps only the latest x years in the set"""

    def __init__(self, data: pd.DataFrame, window_size: int) -> None:
        """
        Args:
        --------------
        data, the original dataframe in RAM
        window_size, the rolling window size, in years
        gpu, gpu or cpu
        """
        self.data = data
        self.gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.window_size = window_size
        self.logger = logging.getLogger("RollingWindow")
        self.window = []

        self.start_year = 1963
        self.start_month = 1

        self.end_year = self.start_year + self.window_size - 1
        self.end_month = 12

        self.start_window = self.start_year * 100 + self.start_month
        self.end_window = self.end_year * 100 + self.end_month

        self.start_end_pairs = {}

        self.train_data = self.build_train_data()

    def build_train_data(self):
        train_data = {}
        start_year = self.start_year
        month = self.start_month
        end_year = 2023

        pairs_window = self.end_window
        while start_year < end_year:
            while month < 13:
                current_month = start_year * 100 + month
                train = self.data[self.data.month == current_month].copy()
                train.drop(columns=["id", "date", "size_grp", "month"], inplace=True)
                r_t_1 = torch.Tensor(train.pop("r_1").values)
                x_t = (
                    torch.Tensor(train.values)
                    .view(train.shape[0], train.shape[1])
                    .to(self.gpu)
                )

                train_data[current_month] = (x_t, r_t_1)
                month += 1

                if current_month <= self.end_window:
                    self.window.append(train_data[current_month])

                self.start_end_pairs[pairs_window] = current_month
                self.end_month += 1

                if self.end_month == 13:
                    self.end_month = 1
                    self.end_year += 1

                pairs_window = self.end_year * 100 + self.end_month

            month = 1
            start_year += 1
        return train_data

    def update_window(self, month: int):
        assert month in self.train_data
        self.start_window = self.start_end_pairs[month]

        self.end_window = month
        self.window.pop(0)
        self.window.append(self.train_data.pop(month))
        self.logger.info(
            f"Start Month:\t{self.start_window}\tEnd Month:\t{self.end_window}\tTotal Months:\t{len(self.window)}"
        )

    #  torch.cuda.empty_cache()
