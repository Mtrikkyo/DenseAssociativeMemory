# import
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd


from tqdm import tqdm
from typing import *


autograd.set_detect_anomaly(True)


# class
class DAMClassifier:
    def __init__(
        self,
        n_visible_unit: int,
        n_class_unit: int,
        n_memory_pattern: int,
        device: torch.device = torch.device("cpu"),
        mu: float | torch.Tensor = -0.3,
        sigma: float = 0.3,
    ) -> None:
        """モデル内の重み行列 weightの作成

        Args:
            n_visible_unit (int): 入力ニューロンの数。 MNISTの場合は784。
            n_class_unit (int): 分類器ニューロンの数。 MNISTの場合には10。
            n_memory_pattern (int): メモリの数。
            mu (float, optional): 重み行列初期化時の正規分布の平均。 デフォルトは -0.3.
            sigma (float, optional): 重み行列初期化時の正規分布の標準偏差。 デフォルトは 0.3.
        """
        # ニューロン数
        self._n_visible_unit = n_visible_unit
        self._n_class_unit = n_class_unit
        self.n_unit = self._n_visible_unit + self._n_class_unit

        # メモリ数
        self.n_memory_pattern = n_memory_pattern

        # デバイス
        self._device = device

        # 重み
        self._weight = nn.Parameter(
            torch.normal(mean=mu, std=sigma, size=(self.n_memory_pattern, self.n_unit), dtype=torch.float64).to(
                self._device
            )
        )

        # クラスニューロンの初期状態

        pass

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        n_epoch: int = 1000,
        optimizer: optim = optim.SGD,
        loss_function=nn.MSELoss,
        energy_function: Callable[[float], float] = lambda x: x**2,
        lr: float = 0.1,
        Temp: float = 540.0,
    ):

        self.energy_function = energy_function

        self.optimizer = optimizer([self._weight], lr=lr)
        self.loss_function = loss_function()

        self.beta = Temp ** (-2)

        for _ in tqdm(range(n_epoch)):

            # train
            for v_unit, label in train_loader:
                v_unit, label = v_unit.to(self._device), label.to(self._device)

                # gradient reset
                self.optimizer.zero_grad()

                batch_size, _ = v_unit.shape  # バッチサイズ

                v_units = torch.tile(
                    v_unit[:, :, None], (1, 1, self._n_class_unit)
                )  # x1 ,x2 ,...x10の更新を同時に行う為の拡張。

                Ux_units = -torch.ones(batch_size, self._n_class_unit, self._n_class_unit).to(
                    self._device
                )  # U行列のxニューロンの状態。
                Vx_units = -torch.ones(batch_size, self._n_class_unit, self._n_class_unit).to(
                    self._device
                )  # V行列のxニューロンの状態。

                for i in range(self._n_class_unit):
                    Vx_units[:, i, i] = -Vx_units[:, i, i]

                U = torch.concatenate([v_units, Ux_units], axis=1).permute(2, 1, 0)
                V = torch.concatenate([v_units, Vx_units], axis=1).permute(2, 1, 0)

                c_unit = self._update_state(U, V)
                loss = self.loss_function(c_unit, label)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self._weight.data = torch.clamp(self._weight.data, min=-1, max=1) # 重みを [-1,1]の範囲にスケーリング


        return

    def predict_proba(self, v_unit):

        v_unit = v_unit.to(self._device)
        batch_size, _ = v_unit.shape
        v_units = torch.tile(
            v_unit[:, :, None], (1, 1, self._n_class_unit)
        )  # x1 ,x2 ,...x10の更新を同時に行う為の拡張。

        Ux_units = -torch.ones(batch_size, self._n_class_unit, self._n_class_unit).to(
            self._device
        )  # U行列のxニューロンの状態。
        Vx_units = -torch.ones(batch_size, self._n_class_unit, self._n_class_unit).to(
            self._device
        )  # V行列のxニューロンの状態。
        for i in range(self._n_class_unit):
            Vx_units[:, i, i] *= -1

        U = torch.concatenate([v_units, Ux_units], axis=1).permute(2, 1, 0)
        V = torch.concatenate([v_units, Vx_units], axis=1).permute(2, 1, 0)

        c_unit = self._update_state(U, V)
        return c_unit

    def _update_state(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """更新則

        Args:
            U (torch.Tensor): U行列。 AppenixAを参照
            V (_type_): V行列。

        Returns:
            _type_: _description_
        """
        logit = self.beta * torch.sum(
            self.energy_function(self._weight @ V) - self.energy_function(self._weight @ U), axis=1
        )

        return torch.softmax(logit, dim=0).T

    @property
    def weight(self):
        return self._weight.cpu().detach()
