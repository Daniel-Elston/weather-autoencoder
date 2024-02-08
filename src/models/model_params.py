from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelParams:
    device: str = 'cpu'
    epochs: int = 10
    lr: float = 0.001
    weight_decay: float = 0.0005
    momentum: float = 0.9
    dampening: float = 0.0


# def main():
#     model_params = ModelParams()
#     print(model_params.__dict__)
#     print(model_params.device)

# if __name__ == "__main__":
#     main()
