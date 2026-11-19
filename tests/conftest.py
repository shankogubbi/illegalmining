"""Keep CPU model checks quick and reproducible across test launch methods."""
import torch


def pytest_sessionstart(session):
    torch.set_num_threads(2)
