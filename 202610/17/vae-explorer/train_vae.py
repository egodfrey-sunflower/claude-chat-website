"""Train a small VAE with a 2D latent space on MNIST and export it for the browser.

Usage (in a uv project with numpy and CPU torch):
    uv run python train_vae.py <mnist_dir> <out_dir> [epochs]

<mnist_dir> holds the four original MNIST .gz files. The architecture follows
Kingma & Welling (2013): an MLP encoder outputs the mean and log-variance of a diagonal
Gaussian q(z|x); z is sampled with the reparameterisation trick; an MLP decoder outputs
Bernoulli pixel probabilities p(x|z); the prior is N(0, I). Training maximises the ELBO.

Writes model.json (architecture, tensor offsets, encoded test points, training stats)
and weights.bin (float16 weights, each stored [in, out] row-major, followed by 200
uint8 test digits).
"""

import gzip
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

H1, H2, Z = 256, 128, 2


def load_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2 = nn.Linear(784, H1), nn.Linear(H1, H2)
        self.mu, self.logvar = nn.Linear(H2, Z), nn.Linear(H2, Z)
        self.dec1, self.dec2, self.dec3 = nn.Linear(Z, H2), nn.Linear(H2, H1), nn.Linear(H1, 784)

    def encode(self, x):
        h = F.relu(self.enc2(F.relu(self.enc1(x))))
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.dec3(F.relu(self.dec2(F.relu(self.dec1(z)))))  # logits

    def loss(self, x):
        """Per-example negative log-likelihood and KL(q(z|x) || N(0, I)), in nats."""
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)  # reparameterisation trick
        nll = F.binary_cross_entropy_with_logits(self.decode(z), x, reduction="none").sum(1)
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
        return nll, kl


def evaluate(model, X, bs=2000):
    with torch.no_grad():
        nll, kl = zip(*(model.loss(X[i:i + bs]) for i in range(0, len(X), bs)))
    nll, kl = torch.cat(nll).mean().item(), torch.cat(kl).mean().item()
    return -(nll + kl), -nll, kl


def main(mnist_dir, out_dir, epochs):
    d = pathlib.Path(mnist_dir)
    xtr = load_idx(d / "train-images-idx3-ubyte.gz").reshape(-1, 784)
    xte = load_idx(d / "t10k-images-idx3-ubyte.gz").reshape(-1, 784)
    yte = load_idx(d / "t10k-labels-idx1-ubyte.gz")
    Xtr = torch.tensor(xtr, dtype=torch.float32) / 255
    Xte = torch.tensor(xte, dtype=torch.float32) / 255

    torch.manual_seed(0)
    model = VAE()
    with torch.no_grad():  # start the decoder at the mean image
        p = Xtr.mean(0).clamp(1e-3, 1 - 1e-3)
        model.dec3.bias.copy_(torch.log(p / (1 - p)))
    bs = 128
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (len(Xtr) // bs), eta_min=1e-4)
    history = []
    for ep in range(1, epochs + 1):
        t0, run = time.time(), 0.0
        perm = torch.randperm(len(Xtr))
        for i in range(len(Xtr) // bs):
            nll, kl = model.loss(Xtr[perm[i * bs:(i + 1) * bs]])
            loss = (nll + kl).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += loss.item()
        tr = -run / (len(Xtr) // bs)
        te, rec, kl = evaluate(model, Xte)
        history.append({"epoch": ep, "train_elbo": round(tr, 2), "test_elbo": round(te, 2)})
        print(f"epoch {ep:3d}  train ELBO {tr:8.2f}  test ELBO {te:8.2f}"
              f"  (recon {rec:.2f}, KL {kl:.2f})  {time.time() - t0:.1f}s", flush=True)
    export(model, xte, yte, Xte, history, (te, rec, kl), pathlib.Path(out_dir))


def export(model, xte, yte, Xte, history, final, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    # Encoder means for a subset of test digits, for the latent-space scatter plot.
    idx = rng.permutation(len(Xte))[:3000]
    with torch.no_grad():
        mu = model.encode(Xte[idx])[0].numpy()
    extent = float(np.ceil(np.percentile(np.abs(mu), 99.5) * 2) / 2)
    # A few real test digits (not in the scatter subset) for the "random digit" button.
    sample_idx = rng.choice(np.setdiff1d(np.arange(len(Xte)), idx), 200, replace=False)

    layers = [("W1", model.enc1), ("W2", model.enc2), ("Wm", model.mu), ("Wv", model.logvar),
              ("D1", model.dec1), ("D2", model.dec2), ("D3", model.dec3)]
    tensors, blobs, off = [], [], 0
    for name, lin in layers:
        W = lin.weight.detach().numpy().T  # torch stores [out, in]; export [in, out]
        b = lin.bias.detach().numpy()
        for n, a in ((name, W), ("b" + name, b)):
            a16 = a.astype(np.float16)
            tensors.append({"name": n, "shape": list(a.shape), "offset": off})
            blobs.append(a16.tobytes()); off += a16.size
    weight_bytes = 2 * off
    blobs.append(xte[sample_idx].astype(np.uint8).tobytes())
    (out / "weights.bin").write_bytes(b"".join(blobs))

    meta = {
        "arch": {"input": 784, "hidden": [H1, H2], "latent": Z,
                 "likelihood": "bernoulli", "prior": "N(0, I)"},
        "tensors": tensors,
        "samples": {"byteOffset": weight_bytes, "count": len(sample_idx),
                    "labels": yte[sample_idx].tolist()},
        "scatter": {"mu": np.round(mu, 3).ravel().tolist(), "labels": yte[idx].tolist()},
        "extent": extent,
        "training": {"dataset": "MNIST", "epochs": len(history), "batch": 128,
                     "optimizer": "Adam, lr 1e-3 cosine-decayed to 1e-4",
                     "test_elbo": round(final[0], 2), "test_recon": round(final[1], 2),
                     "test_kl": round(final[2], 2), "history": history},
    }
    (out / "model.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"exported to {out}: extent ±{extent}, weights {weight_bytes / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 50)
