import torch
import torch.nn as nn


class PropulsionLinear(nn.Module):
    def __init__(
        self, input_features, output_features, bias=False, degree=15, **kwargs
    ):
        super(PropulsionLinear, self).__init__()
        self.prop_linear = nn.Linear(
            input_features, output_features, bias=bias, **kwargs
        )
        self.propulsion = nn.Parameter(torch.ones(output_features))
        self.degree = degree

    def forward(self, x):
        self.push = torch.pow(self.propulsion, self.degree)
        return self.prop_linear(x) * self.push
