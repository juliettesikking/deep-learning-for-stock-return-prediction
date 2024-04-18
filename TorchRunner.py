import time
import torch
import os
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from RollingWindow import RollingWindow
import numpy as np
import logging


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
        rolling_window: bool,
        window_size: int,
        end_year: int,
        output: str,
        epochs: int,
        data: pd.DataFrame,
        resume: bool,
        no_incremental: bool,
        model_name: str,
        coordinate_check: bool = False,
    ) -> None:
        self.rolling_window = rolling_window
        self.window_size = window_size
        self.start_year = 1963 + window_size if rolling_window else 1963
        self.month = 1 if rolling_window else 2
        self.end_year = end_year
        self.output = output
        self.data = data
        self.epochs = epochs
        self.resume = resume
        self.no_incremental = no_incremental
        self.gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model_name
        self.coordinate_check = coordinate_check

    @staticmethod
    def run_epoch(data_iter, model, loss_compute, optimizer, coordinate_check: bool):
        """Train a single epoch"""
        start = time.time()
        total_loss = 0
        n_accum = 0
        in_sample_results = []
        # Batches
        for i, month_data in enumerate(data_iter):
            x_t, r_t_1 = month_data

            out, norms = model.forward(x_t)
            if coordinate_check and not (
                abs(norms[-1]) < abs(norms[0]) + (abs(norms[0]) * 0.2)
            ):
                return abs(norms[-1]) < abs(norms[0]) + (abs(norms[0]) * 0.2)
            out = out.cpu()

            loss = loss_compute(out, r_t_1.view(-1, 1))

            in_sample_results += [torch.sum(out * r_t_1.view(-1, 1))]

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1

            total_loss += float(loss)
            if (len(data_iter) > 10 and i % 10 == 0) or (i == (len(data_iter) - 1)):
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                logging.info(
                    (
                        "Epoch Step: %6d | num. month run with GD: %3d | Loss for one month: %6.5f "
                        + " | Learning Rate: %6.1e | Time: %6.2fs"
                    )
                    % (i, n_accum, loss / x_t.shape[0], lr, elapsed)
                )
                start = time.time()

            torch.cuda.empty_cache()
        in_sample_results = torch.stack(in_sample_results)
        in_sample_sharpe = (
            np.sqrt(12) * in_sample_results.mean() / in_sample_results.std()
        )
        in_sample_sharpe = in_sample_sharpe.item()
        logging.info(f"total average loss after this epoch is: {total_loss / n_accum}")
        logging.info(f"in-sample sharpe: {in_sample_sharpe}")
        logging.info(f"norms of hidden layers: {norms}")
        if coordinate_check:
            return True
        return in_sample_sharpe

    def run(self,  model: torch.nn.Module, criterion, optimizer, window: RollingWindow):
        previous_month = self.start_year * 100 + 1

        pf_returns = dict()
        in_sample_sharpes = dict()
        start = time.time()
        while self.start_year < self.end_year:
            while self.month < 13:
                current_month = self.start_year * 100 + self.month
                if current_month == 202212:
                    break

                if not self.rolling_window:
                    window.update_window(previous_month)

                test = self.data[self.data.month == current_month].copy()
                test_info = self.data[self.data.month == current_month][
                    ["id", "date", "size_grp", "r_1"]
                ].copy()

                # train.drop(columns=["id", "date", "size_grp", "month"], inplace=True)
                test.drop(
                    columns=["id", "date", "size_grp", "month", "r_1"], inplace=True
                )

                test = torch.Tensor(test.values).view(1, test.shape[0], test.shape[1])

                if (self.resume > 0) and current_month <= self.resume:
                    previous_month = current_month
                    self.month += 1
                    if self.rolling_window:
                        window.update_window(month=current_month)
                    continue

                # ******************
                # TrainData
                # ******************

                dataloader = DataLoader(window.window, shuffle=False, batch_size=1)
                # ******************
                # Training
                # ******************

                if self.no_incremental:
                    reset_param(model)

                for epoch in range(self.epochs):
                    model.train()
                    logging.info(
                        f"{'='*12}>[GPU{self.gpu}]\tEpoch\t{epoch}\tTest Month\t{current_month}===="
                    )
                    if self.coordinate_check:
                        is_stable = TorchRunner.run_epoch(
                            dataloader,
                            model,
                            criterion,
                            optimizer,
                            self.coordinate_check,
                        )

                        # Not stable end right away
                        if not is_stable:
                            return is_stable

                    else:
                        in_sample_sharpe = TorchRunner.run_epoch(
                            dataloader,
                            model,
                            criterion,
                            optimizer,
                            coordinate_check=False
                        )

                if not self.coordinate_check:
                    model.eval()
                    with torch.no_grad():
                        predictions, norms = model(test.to(self.gpu))
                    logging.info(
                        f"norms of hidden layers after one epoch cycle: {norms}"
                    )
                    test_info["predictions"] = predictions.detach().cpu()
                    test_info["in_sample_sharpe"] = in_sample_sharpe

                    del dataloader
                    torch.cuda.empty_cache()

                    # save everything
                    # XXX fix this
                    pf_returns[shift_one_month(current_month)] = sum(
                        test_info.r_1 * test_info.predictions
                    )
                    in_sample_sharpes[current_month] = in_sample_sharpe

                    # The model is trained using X_{t-T}, ..., X_{t-1}
                    # Then we make a predictions using X_t
                    # Thus \tilde X = (\sum_h X_t M X_t') X_t
                    if self.model == "interpretable" or self.model.startswith("factor"):
                        epoch_dir = os.path.join(self.output, str(current_month))
                        os.makedirs(epoch_dir, exist_ok=True)
                        torch.save(
                            model.tilde_X.detach().cpu(),
                            os.path.join(epoch_dir, "tilde_X.pt"),
                        )
                previous_month = current_month
                self.month += 1
                if self.rolling_window:
                    window.update_window(month=current_month)
            self.month = 1
            self.start_year += 1

            if self.coordinate_check:
                return is_stable


        end = time.time()
        logging.info(f"Total Training time:\t{end-start:.2f}")

        pd.Series(pf_returns).to_pickle(os.path.join(self.output, "returns.pickle"))
        pd.Series(in_sample_sharpes).to_pickle(
            os.path.join(self.output, "in_sample_sharpes.pickle")
        )
