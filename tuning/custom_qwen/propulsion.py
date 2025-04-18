import torch
import torch.nn as nn
import torch.nn.functional as F


class PropulsionLinear(nn.Module):
    def __init__(
        self, input_features, output_features, bias=False, degree=15, **kwargs
    ):
        super(PropulsionLinear, self).__init__()
        linear = nn.Linear(input_features, output_features, bias=bias, **kwargs)
        self.propulsion = nn.Parameter(torch.ones(output_features), requires_grad=True)

        weight = linear.weight
        bias_weight = linear.bias if bias else None

        weight.requires_grad = False

        if bias_weight is not None:
            bias_weight.requires_grad = True

        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias_weight) if bias_weight is not None else None
        )

        self.degree = degree

    def forward(self, x):
        self.push = torch.pow(self.propulsion, self.degree)
        return F.linear(x, self.weight, self.bias) * self.push
