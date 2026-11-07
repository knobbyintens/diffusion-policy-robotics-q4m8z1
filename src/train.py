import argparse
import torch
from torch.utils.data import DataLoader

from .model import DiffusionPolicy


def train(dataset, epochs=100, lr=1e-4):
    model = DiffusionPolicy(dataset.obs_dim, dataset.act_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for obs, actions in loader:
            noise = torch.randn_like(actions)
            loss = ((model(obs, noise) - actions.flatten(1)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    train(None, epochs=parser.parse_args().epochs)
