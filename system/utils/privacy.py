try:
    from opacus import PrivacyEngine
except ImportError:
    PrivacyEngine = None
import torch
import torch.nn.functional as F


MAX_GRAD_NORM = 1.0
DELTA = 1e-5

def initialize_dp(model, optimizer, data_loader, dp_sigma):
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier = dp_sigma, 
        max_grad_norm = MAX_GRAD_NORM,
    )

    return model, optimizer, data_loader, privacy_engine


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA

def cos(t1, t2):
    t1 = F.normalize(t1, dim=0)
    t2 = F.normalize(t2, dim=0)

    dot = (t1 * t2).sum(dim=0)

    return dot

def pair_cos(pair):
    length = pair.size(0)

    dot_value = []
    for i in range(length - 1):
        for j in range(i + 1, length):
            dot_value.append(cos(pair[i], pair[j]))

    dot_value = torch.stack(dot_value).view(-1)
    return dot_value