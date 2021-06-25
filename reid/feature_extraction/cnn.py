from __future__ import absolute_import
from collections import OrderedDict

import torch

from ..utils import to_torch

def extract_cnn_feature(model, inputs, sub, modules=None):
    model.eval()
    inputs = to_torch(inputs)
    inputs = inputs.cuda()
    with torch.no_grad():
        if modules is None:
            # whether "modules" is None or not, this function is used to extract feature from a certain dataset.
            #_, outputs = model(inputs)
            i_observation, i_representation, i_ms_observation, i_ms_representation, \
            v_observation, v_representation, v_ms_observation, v_ms_representation = model(inputs)
            outputs = torch.cat(tensors=[i_observation[1], i_ms_observation[1], i_representation[1], i_ms_representation[1],
                                         v_observation[1], v_ms_observation[1], v_representation[1], v_ms_representation[1]], dim=1)
            outputs = outputs.data.cpu()
            return outputs
        # Register forward hook for each module
        outputs = OrderedDict()
        handles = []
        for m in modules:
            outputs[id(m)] = None
            def func(m, i, o): outputs[id(m)] = o.data.cpu()
            handles.append(m.register_forward_hook(func))
        model(inputs)
        for h in handles:
            h.remove()
        return list(outputs.values())
