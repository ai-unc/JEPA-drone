import argparse
from pathlib import Path

import torch

from skyjepa.config import Config
from skyjepa.models import encoders, predictor


def load_jepa(path: Path, device: str = "cpu"):
    """Rebuild the encoders and predictor, load their weights, and switch to eval."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = Config(**checkpoint["config"])
    state_encoder = encoders.StateHistoryEncoder(config).to(device)
    action_encoder = encoders.ActionHistoryEncoder(config).to(device)
    latent_predictor = predictor.LatentPredictor(
        latent_dim=config.latent_dim,
        action_dim=config.latent_dim,
        hidden_dim=config.latent_dim,
    ).to(device)
    state_encoder.load_state_dict(checkpoint["state_encoder"])
    action_encoder.load_state_dict(checkpoint["action_encoder"])
    latent_predictor.load_state_dict(checkpoint["predictor"])
    state_encoder.eval()
    action_encoder.eval()
    latent_predictor.eval()
    return state_encoder, action_encoder, latent_predictor, checkpoint["norm"], config


def main() -> None:
    
    parser = argparse.ArgumentParser(description="Load a trained JEPA checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/jepa.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    state_encoder, action_encoder, latent_predictor, norm, config = load_jepa(
        args.checkpoint, args.device,
    )
    print(
        f"loaded {args.checkpoint} on {args.device}: "
        f"latent_dim={config.latent_dim} history={config.history_len} horizon={config.horizon}"
    )
    print(
        "eval: "
        f"state={not state_encoder.training} "
        f"action={not action_encoder.training} "
        f"predictor={not latent_predictor.training}"
    )
    print("normalizer keys:", ", ".join(norm))


if __name__ == "__main__":
    main()
