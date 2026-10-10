import numpy as np
import torch

from .data.dataset import REFERENCE, TRAIN_DEVICES, UNSEEN_DEVICES


@torch.no_grad()
def predict(model, loader, garden, device):
    was = model.training
    model.eval()
    ref = TRAIN_DEVICES.index(REFERENCE)
    out = []
    for x, _, d in loader:
        x, d = x.to(device), d.to(device)
        out.append((model(x, d == ref) if garden else model(x)).float().cpu())
    model.train(was)
    return torch.cat(out)


def scores(pred, labels, devices):
    correct = pred == labels
    acc = {str(d): 100 * float(correct[devices == d].mean()) for d in np.unique(devices)}
    known = float(np.mean([acc[d] for d in TRAIN_DEVICES if d in acc]))
    unseen = float(np.mean([acc[d] for d in UNSEEN_DEVICES if d in acc]))
    macro = float(np.mean(list(acc.values())))
    return {"known": known, "unseen": unseen, "gap": known - unseen, "macro": macro, **acc}
