import time
import torch
import os
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from RollingWindow import RollingWindow
import numpy as np
import logging
from pilimit_lib.inf.optim import PiSGD, store_pi_grad_norm_, clip_grad_norm_

logging.basicConfig(level=logging.INFO)


def shift_one_month(date):
    year = date // 100
    month = date % 100
    month = (month % 12) + 1
    year += int(month == 1)
    result_date = year * 100 + month
    return result_date


class TorchRunner:
    """This class runs a generic torch Model"""

    def __init__(
        self,
        window_size: int,
        end_year: int,
        output: str,
        epochs: int,
        data: pd.DataFrame,
        resume: bool,
        no_incremental: bool,
        model_name: str,
        coordinate_check: bool = False,
    ):

        self.window_size = window_size
        self.start_year = 1963 + window_size
        self.month = 1
        self.end_year = end_year
        self.output = output
        self.data = data
        self.epochs = epochs
        self.resume = resume
        self.no_incremental = no_incremental
        self.gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model_name
        self.coordinate_check = coordinate_check

    def run(self, model: torch.nn.Module, criterion, optimizer, window: RollingWindow):
        previous_month = self.start_year * 100 + 1
        pf_returns = {}
        in_sample_sharpes = {}
        start = time.time()

        while self.start_year < self.end_year:
            while self.month < 13:
                current_month = self.start_year * 100 + self.month
                if current_month == 202212:
                    break

                test = self.data[self.data["month"] == current_month].copy()
                test_info = self.data[self.data["month"] == current_month][
                    ["id", "date", "size_grp", "r_1"]
                ].copy()
                test.drop(
                    columns=["id", "date", "size_grp", "month", "r_1"], inplace=True
                )
                test_tensor = (
                    torch.Tensor(test.values)
                    .view(1, test.shape[0], test.shape[1])
                    .to(self.gpu)
                )

                dataloader = DataLoader(window.window, shuffle=False, batch_size=1)

                for epoch in range(self.epochs):
                    model.train()
                    logging.info(
                        f"{'='*12}>[GPU{self.gpu}]\tEpoch\t{epoch}\tTest Month\t{current_month}===="
                    )
                    self.run_epoch(
                        dataloader,
                        model,
                        criterion,
                        optimizer,
                    )

                model.eval()
                with torch.no_grad():
                    predictions = model(test_tensor)
                    predictions = predictions.view(-1)
                    test_info["predictions"] = predictions.detach().cpu()
                pf_returns[shift_one_month(current_month)] = sum(
                    test_info.r_1 * test_info.predictions
                )
                # Save and log results as needed
                previous_month = current_month
                self.month += 1
                window.update_window(month=current_month)
            self.month = 1
            self.start_year += 1

        end = time.time()
        logging.info(f"Total Training time: {end-start:.2f}s")
        returns = pd.Series(pf_returns)
        returns.to_pickle(os.path.join(self.output, "returns.pickle"))
        print(f"Sharpe: {returns.mean()/returns.std() * np.sqrt(12)}")
        print(returns.mean)
        pd.Series(in_sample_sharpes).to_pickle(
            os.path.join(self.output, "in_sample_sharpes.pickle")
        )
        return returns

    def run_epoch(self, data_iter, model, loss_compute, optimizer):
        start = time.time()
        total_loss = 0
        n_accum = 0
        in_sample_results = []
        for i, (x_t, r_t_1) in enumerate(data_iter):
            out = (
                model(x_t)
                if self.model != "pilimit"
                else model(x_t.view(x_t.size(1), x_t.size(2)))
            )
            loss = loss_compute(out, r_t_1.view(-1, 1))
            loss.backward()

            if self.model == "pilimit":
                store_pi_grad_norm_(model.modules())
                clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            total_loss += float(loss)
            in_sample_results += [torch.sum(out * r_t_1.view(-1, 1))]
            if (len(data_iter) > 10 and i % 10 == 0) or (i == (len(data_iter) - 1)):
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                logging.info(
                    f"Epoch Step: {i} | num. month run with GD: {n_accum} | Loss for one month: {loss / x_t.shape[0]:.5f} | Learning Rate: {lr:.1e} | Time: {elapsed:.2f}s"
                )
                start = time.time()

        in_sample_results = torch.stack(in_sample_results)
        in_sample_sharpe = (
            np.sqrt(12) * in_sample_results.mean() / in_sample_results.std()
        )
        in_sample_sharpe = in_sample_sharpe.item()
        logging.info(f"total average loss after this epoch is: {total_loss / n_accum}")
        logging.info(f"in-sample sharpe: {in_sample_sharpe}")
