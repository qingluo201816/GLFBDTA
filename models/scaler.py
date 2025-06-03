from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.utils import degree


class DegreeScalerAggregation(Aggregation):

    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation],
        scaler: Union[str, List[str]],
        deg: Tensor,
        aggr_kwargs: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()

        if isinstance(aggr, (str, Aggregation)):
            self.aggr = aggr_resolver(aggr, **(aggr_kwargs or {}))
        elif isinstance(aggr, (tuple, list)):
            self.aggr = MultiAggregation(aggr, aggr_kwargs)
        else:
            raise ValueError(f"Only strings, list, tuples and instances of"
                             f"`torch_geometric.nn.aggr.Aggregation` are "
                             f"valid aggregation schemes (got '{type(aggr)}')")

        self.scaler = [scaler] if isinstance(aggr, str) else scaler

        deg = deg.to(torch.float)
        num_nodes = int(deg.sum())
        bin_degrees = torch.arange(deg.numel(), device=deg.device)
        self.avg_deg: Dict[str, float] = {
            'lin': float((bin_degrees * deg).sum()) / num_nodes,
            'log': float(((bin_degrees + 1).log() * deg).sum()) / num_nodes,
            'exp': float((bin_degrees.exp() * deg).sum()) / num_nodes,
        }

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,

                dim: int = -2) -> Tensor:


        self.assert_index_present(index)

        out = self.aggr(x, index, ptr, dim_size, dim)

        assert index is not None
        deg = degree(index, num_nodes=dim_size, dtype=out.dtype).clamp_(1)
        size = [1] * len(out.size())
        size[dim] = -1
        deg = deg.view(size)

        outs = []
        for scaler in self.scaler:
            if scaler == 'identity':
                out_scaler = out
            elif scaler == 'amplification':
                out_scaler = out * (torch.log(deg + 1) / self.avg_deg['log'])

            elif scaler == 'attenuation':
                out_scaler = out * (self.avg_deg['log'] / torch.log(deg + 1))

            elif scaler == 'exponential':
                out_scaler = out * (torch.exp(deg) / self.avg_deg['exp'])

            elif scaler == 'linear':
                out_scaler = out * (deg / self.avg_deg['lin'])

            elif scaler == 'inverse_linear':
                out_scaler = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f"Unknown scaler '{scaler}'")
            outs.append(out_scaler)

        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
