from collections import OrderedDict
from itertools import chain, starmap
from typing import Iterable, Tuple, Type

import torch


class DenseNet(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        n_classes: int,
        n_layers: int,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
    ):
        '''
        :param int in_features: Число входных признаков
        :param int hidden_size: Размер скрытых слоёв
        :param int n_classes: Число выходов сети
        :param int n_layers: Число слоёв в сети
        :param torch.nn.Module activation: Класс функции активации
        '''
        super().__init__()

        in_sizes = [in_features] + [hidden_size] * (n_layers - 1)
        out_sizes = [hidden_size] * (n_layers - 1) + [n_classes]
        add_activation = [True] * (n_layers - 1) + [False]

        def layer_generator(
            n_layer: int, in_size: int, out_size: int, add_activation_: bool
        ) -> Iterable[Tuple[str, torch.nn.Module]]:
            layer_name = f'{torch.nn.Linear.__name__}_layer:{n_layer}'
            yield layer_name, torch.nn.Linear(in_features=in_size, out_features=out_size)
            if add_activation_:
                yield f'{activation.__name__}_layer:{n_layer}', activation()

        layer_generators = starmap(
            layer_generator, zip(range(n_layers), in_sizes, out_sizes, add_activation)
        )
        layers = chain.from_iterable(layer_generators)
        self.layers = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Прямой проход по сети
        :param torch.Tensor x: Входной тензор размера [batch_size, in_features]
        :returns: Матрица логитов размера [batch_size, n_classes]
        '''
        if x.dtype == torch.double:
            x = x.to(torch.float)
        return self.layers(x)
