import torch
from torch import nn
import json
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from skyjepa.config import Config

CLASSES = {
    "unknown": 0,
    "wall": 1,
    "floor": 2,
    "ceiling": 3,
    "door": 4,
    "window": 5,
    "furniture": 6,
    "obstacle": 7,
}

class StateHistoryEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.tcn = nn.Sequential(
            nn.Conv1d(config.state_dim, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(16, config.latent_dim)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """(B, H, state_dim) -> (B, latent_dim)."""
        x = state_history.transpose(1, 2)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.projection(x)


class ActionHistoryEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.tcn = nn.Sequential(
            nn.Conv1d(config.action_dim, 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(4, 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=3, padding=1), nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(8, config.latent_dim)

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """(B, H, action_dim) -> (B, latent_dim)."""
        x = action_history.transpose(1, 2)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.projection(x)


class ProprioceptiveEncoder(nn.Module):

    def __init__(self, state_dim: int = 18, feature_dim: int = 32, hidden_dim: int = 64, latent_dim: int = 32, history_length: int = 10,):
        super().__init__()

        self.state_dim = state_dim
        self.history_length = history_length

        # extract features from individual timesteps
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, feature_dim),
            nn.GELU(),
        )

        # capture temporal movement dynamics
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # project into proprioceptive latent space
        self.projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:

        if states.ndim != 3 or states.shape[1:] != (self.history_length, self.state_dim):
            raise ValueError("Invalid state history shape")

        # express positions relative to the latest position
        relative_pos = states[:, :, :3] - states[:, -1:, :3]

        states = torch.cat([relative_pos, states[:, :, 3:]], dim=-1)

        # per-timestep feature extraction
        features = self.mlp(states)

        # temporal encoding
        _, hidden = self.gru(features)

        # project final hidden state
        latent = self.projection(hidden[-1])

        return latent


class ImageEncoder(nn.Module):
    """image encoder for AirSim only (not to be used for IRL testing)"""

    def __init__(self, latent_dim=32, num_classes=8):
        super().__init__()

        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(1 + num_classes, 16, 3, 2, 1),
            nn.GELU(),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.GELU(),

            nn.Conv2d(32, 32, 3, 2, 1),
            nn.GELU()
            )

        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(32 * 8 * 8, latent_dim),
        )

    def forward(self, depth, segmentation):

        # depth: (B, 1, 64, 64)
        # segmentation: (B, 64, 64)

        # convert labels to binary channels
        semantic = F.one_hot(
            segmentation.long(),
            num_classes=self.num_classes,)

        semantic = semantic.permute(0, 3, 1, 2)
        semantic = semantic.to(depth.dtype)

        # combine depth and semantic information
        images = torch.cat(
            [depth, semantic],
            dim=1)

        # CNN feature extraction
        features = self.cnn(images)

        # project into latent space
        latent = self.projection(features)

        return latent


class TextEncoder(nn.Module): 
    
    INTENTS = [
        "move_relative",
        "move_absolute",
        "exit_room",
        "hover",
        "land",
        "explore",
        "approach_object",
        "other"
    ]

    OBJECTS = [
        "unknown",
        "wall",
        "floor",
        "ceiling",
        "door",
        "window",
        "furniture",
        "obstacle"
    ]

    def __init__(self, device="cpu"):
        super().__init__()

        model_name = "Qwen/Qwen3-0.6B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        self.model.requires_grad_(False)
        self.model.eval()

        self.semantic_dim = self.model.config.hidden_size
        self.goal_dim = 23 + self.semantic_dim

    @torch.no_grad()
    def parse_goal(self, command):

        prompt = f"""
        Parse the natural language drone command.

        Available intents: {self.INTENTS}
        Available objects: {self.OBJECTS}

        Return only a JSON object with these fields:

        {{
            "intent": "intent_name",
            "object": "object_name",
            "target": null,
            "preference": "none"
        }}

        Rules:
        - Target is [x, y, z] in meters or null.
        - Do not invent spatial coordinates.
        - Relative coordinates: X forward, Y right, Z down.
        - Absolute coordinates: AirSim world NED frame.
        - Preference must be nearest, farthest, or none.
        - Use unknown when no object is specified.
        - Use other for unsupported or compound objectives.
        - For exiting a room, the target object is door.
        - Do not generate intermediate waypoints.
        """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": command},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20
            )

        generated = output[0, inputs.input_ids.shape[1]:]

        response = self.tokenizer.decode(generated, skip_special_tokens=False)
        if "</think>" in response:
            response = response.split("</think>")[-1]
        response = response.replace("<|im_end|>", "").strip()

        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"Model did not return JSON:\n{response}")
        
        return json.loads(response[start:end + 1])

    def encode_structured_goal(self, goal):

        tensor = torch.zeros(
            23,
            device=self.model.device,
        )

        intent = goal["intent"]
        obj = goal["object"]
        target = goal["target"]
        preference = goal["preference"]

        if intent not in self.INTENTS:
            raise ValueError("Invalid intent")

        if obj not in self.OBJECTS:
            raise ValueError("Invalid object")

        if preference not in ("nearest", "farthest", "none"):
            raise ValueError("Invalid preference")

        # One-hot encode intent
        tensor[self.INTENTS.index(intent)] = 1.0

        # One-hot encode object
        tensor[8 + self.OBJECTS.index(obj)] = 1.0

        # encode coordinates when available
        if target is not None:
            if (
                not isinstance(target, list)
                or len(target) != 3
                or not all(
                    type(v) in (int, float)
                    and math.isfinite(v)
                    for v in target
                )
            ):
                raise ValueError("Invalid coordinates")

            tensor[16:19] = torch.tensor(
                target,
                device=tensor.device,
            )
            tensor[19] = 1.0

        elif intent in ("move_relative", "move_absolute"):
            raise ValueError("Movement target missing")

        # object selection preferences
        tensor[20] = float(preference == "nearest")
        tensor[21] = float(preference == "farthest")

        # default position tolerance
        tensor[22] = 0.2

        return tensor.unsqueeze(0)

    @torch.no_grad()
    def encode_semantics(self, command):

        inputs = self.tokenizer(
            command,
            return_tensors="pt",
        ).to(self.model.device)

        output = self.model.base_model(
            **inputs,
            use_cache=False,
        )

        hidden = output.last_hidden_state

        # average representations across tokens
        embedding = hidden.mean(dim=1)

        # normalize embedding
        embedding = F.normalize(
            embedding.float(),
            dim=-1,
        )

        return embedding

    def forward(self, command):

        parsed = self.parse_goal(command)

        structured = self.encode_structured_goal(parsed)
        semantic = self.encode_semantics(command)

        goal_tensor = torch.cat(
            [structured, semantic],
            dim=-1,
        )

        return goal_tensor, parsed


class MultimodalEncoder(nn.Module):

    def __init__(self, input_dim: int = 96):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )

    def forward(self, image_tensor: torch.Tensor, proprioceptive_tensor: torch.Tensor, state_history: torch.Tensor):
        combined = torch.cat([image_tensor, proprioceptive_tensor, state_history], dim=-1)

        return self.mlp(combined)
