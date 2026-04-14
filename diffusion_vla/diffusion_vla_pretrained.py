import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import CLIPTextModel, CLIPTokenizer

try:
    from torchvision import models
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
    _HAS_TORCHVISION = True
except Exception:
    models = None
    ResNet18_Weights = None
    ResNet34_Weights = None
    ResNet50_Weights = None
    _HAS_TORCHVISION = False


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


# ============================================================
# CLIP text processor / text encoder
# ============================================================

class CLIPTextProcessor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", max_length: int = 48):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        out = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": out["input_ids"].squeeze(0),
            "attention_mask": out["attention_mask"].squeeze(0),
        }


class CLIPTextEncoder(nn.Module):
    """
    Text: CLIP tokenizer + CLIPTextModel
    Output: projected to d_model
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        out_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = False,
        pooling: str = "mean",
    ):
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling

        if pretrained:
            self.clip_text = CLIPTextModel.from_pretrained(model_name)
        else:
            self.clip_text = CLIPTextModel.from_pretrained(model_name)

        clip_dim = self.clip_text.config.hidden_size
        self.proj = nn.Linear(clip_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        if freeze:
            freeze_module(self.clip_text)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.clip_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state  # [B, T, C]

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        elif self.pooling == "eos":
            # eos token 위치를 attention_mask 길이 기준으로 잡음
            lengths = attention_mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        pooled = self.proj(pooled)
        return self.norm(pooled)


# ============================================================
# LoRA
# ============================================================

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(self.dropout(x), self.B @ self.A, None) * self.scaling
        return base + update


_PAT = re.compile(r"(q_proj|k_proj|v_proj|out_proj|fc1|fc2|cond_proj|time_proj|proj)$")


def attn_name_filter(full_name: str, sub: nn.Module) -> bool:
    return isinstance(sub, nn.Linear) and bool(_PAT.search(full_name.split(".")[-1]))


def replace_linear_with_lora(module: nn.Module, name_filter, r: int = 8, alpha: int = 8,
                             dropout: float = 0.0, prefix: str = "") -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full = f"{prefix}.{child_name}" if prefix else child_name
        replaced += replace_linear_with_lora(child, name_filter, r=r, alpha=alpha, dropout=dropout, prefix=full)
        if name_filter(full, child) and not isinstance(child, LoRALinear):
            new = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            new.train(child.training)
            try:
                p = next(child.parameters())
                new = new.to(device=p.device, dtype=p.dtype)
            except StopIteration:
                pass
            setattr(module, child_name, new)
            replaced += 1
    return replaced


def enable_lora_only(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True


def lora_parameters(module: nn.Module):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield m.A
            yield m.B


def state_dict_lora_only(module: nn.Module) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            sd[f"{name}.A"] = m.A.detach().cpu()
            sd[f"{name}.B"] = m.B.detach().cpu()
            sd[f"{name}.scaling"] = torch.tensor(float(m.scaling))
    return sd


def load_state_dict_lora_only(module: nn.Module, lora_sd: Dict[str, torch.Tensor]) -> None:
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            if f"{name}.A" in lora_sd:
                m.A.data.copy_(lora_sd[f"{name}.A"])
            if f"{name}.B" in lora_sd:
                m.B.data.copy_(lora_sd[f"{name}.B"])
            if f"{name}.scaling" in lora_sd:
                m.scaling = float(lora_sd[f"{name}.scaling"])


# ============================================================
# DDPM scheduler
# ============================================================

class DDPMScheduler:
    def __init__(self, num_train_steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_steps, dtype=torch.float32) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.num_train_steps = num_train_steps
        self.register(betas, alphas, alphas_cumprod)

    def register(self, betas: torch.Tensor, alphas: torch.Tensor, alphas_cumprod: torch.Tensor) -> None:
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

    def to(self, device: torch.device) -> "DDPMScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[timesteps].view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, generator=None) -> torch.Tensor:
        beta_t = self.betas[timestep]
        alpha_t = self.alphas[timestep]
        alpha_bar_t = self.alphas_cumprod[timestep]
        alpha_bar_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0, device=sample.device)

        pred_x0 = (sample - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
        coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        mean = coef_x0 * pred_x0 + coef_xt * sample

        if timestep == 0:
            return mean

        variance = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
        noise = torch.randn(sample.shape, device=sample.device, generator=generator)
        return mean + torch.sqrt(variance.clamp_min(1e-12)) * noise


# ============================================================
# Dataset
# ============================================================

def _flatten_float_list(x) -> List[float]:
    if isinstance(x, list):
        out: List[float] = []
        for v in x:
            if isinstance(v, list):
                out.extend(_flatten_float_list(v))
            else:
                out.append(float(v))
        return out
    return [float(x)]


def build_state_vector(row: Dict) -> List[float]:
    state = row.get("state", {})
    obs = row.get("observation", {})

    state_vec = []
    if "arm_joint_position" in state:
        state_vec.extend(_flatten_float_list(state["arm_joint_position"]))
    if "gripper_width" in state:
        state_vec.extend(_flatten_float_list(state["gripper_width"]))
    if "ee_pose" in state:
        state_vec.extend(_flatten_float_list(state["ee_pose"]))

    if not state_vec and isinstance(obs, dict):
        for k in ["state", "EEF_state", "gripper_state"]:
            if k in obs:
                state_vec.extend(_flatten_float_list(obs[k]))

    return state_vec


def build_action_vector(action: Dict, action_format: str = "auto") -> Tuple[List[float], str]:
    resolved_format = action_format
    if resolved_format == "auto":
        if any(k in action for k in ["world_vector", "rotation_delta"]):
            resolved_format = "cartesian_delta"
        elif any(k in action for k in ["joint_delta", "arm_joint_delta"]):
            resolved_format = "joint_delta"
        else:
            raise ValueError(f"Could not infer action format from keys: {sorted(action.keys())}")

    action_vec = []
    if resolved_format == "cartesian_delta":
        if "world_vector" in action:
            action_vec.extend(_flatten_float_list(action["world_vector"]))
        if "rotation_delta" in action:
            action_vec.extend(_flatten_float_list(action["rotation_delta"]))
        if "gripper_closedness_action" in action:
            action_vec.extend(_flatten_float_list(action["gripper_closedness_action"]))
    elif resolved_format == "joint_delta":
        if "joint_delta" in action:
            action_vec.extend(_flatten_float_list(action["joint_delta"]))
        elif "arm_joint_delta" in action:
            action_vec.extend(_flatten_float_list(action["arm_joint_delta"]))
        else:
            raise ValueError("joint_delta action format requires 'joint_delta' or 'arm_joint_delta'")

        if "gripper_delta" in action:
            action_vec.extend(_flatten_float_list(action["gripper_delta"]))
        elif "gripper_width_delta" in action:
            action_vec.extend(_flatten_float_list(action["gripper_width_delta"]))
        elif "gripper_action" in action:
            action_vec.extend(_flatten_float_list(action["gripper_action"]))
        elif "gripper_closedness_action" in action:
            action_vec.extend(_flatten_float_list(action["gripper_closedness_action"]))
        else:
            raise ValueError("joint_delta action format requires gripper delta/action field")
    else:
        raise ValueError(f"Unsupported action format: {resolved_format}")

    terminate = action.get("terminate_episode", 0.0)
    action_vec.extend(_flatten_float_list(terminate))
    return action_vec, resolved_format


def format_output_action(action_vec: Sequence[float], action_format: str) -> Dict:
    values = [float(v) for v in action_vec]
    if action_format == "joint_delta":
        if len(values) < 9:
            raise ValueError(f"joint_delta output expects at least 9 dims, got {len(values)}")
        return {
            "joint_delta": values[:7],
            "gripper_delta": [values[7]],
            "terminate_episode": float(values[8]),
        }

    if len(values) < 8:
        raise ValueError(f"cartesian_delta output expects at least 8 dims, got {len(values)}")
    return {
        "world_vector": values[:3],
        "rotation_delta": values[3:6],
        "gripper_closedness_action": [values[6]],
        "terminate_episode": float(values[7]),
    }


@dataclass
class RobotFrame:
    image_path: str
    instruction: str
    state: List[float]
    action: List[float]
    episode_key: str
    frame_index: int


class OpenVLADeltaDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str, text_processor: CLIPTextProcessor,
                 image_size: int = 224, horizon: int = 4,
                 action_dim: int = 0, normalize: bool = True, action_format: str = "auto"):
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.text_processor = text_processor
        self.image_size = image_size
        self.horizon = horizon
        self.action_dim = action_dim
        self.normalize = normalize
        self.action_format = action_format

        self.frames = self._load_frames(jsonl_path)
        self.samples = self._build_sequence_indices(self.frames, horizon)

        self.state_dim = len(self.frames[0].state)
        self.action_stats = self._compute_action_stats() if normalize else None
        self.state_stats = self._compute_state_stats() if normalize else None

    def _resolve_image_path(self, rel_or_abs: str) -> str:
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.join(self.image_root, rel_or_abs)

    def _load_frames(self, path: str) -> List[RobotFrame]:
        frames: List[RobotFrame] = []
        inferred_action_format: Optional[str] = None
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                obs = row.get("observation", {})
                state = row.get("state", {})
                action = row.get("action", {})

                image_path = self._resolve_image_path(obs.get("image", ""))
                instruction = obs.get("natural_language_instruction") or obs.get("instruction") or "do the task"

                state_vec = build_state_vector(row)
                action_vec, row_action_format = build_action_vector(action, self.action_format)
                if inferred_action_format is None:
                    inferred_action_format = row_action_format
                elif row_action_format != inferred_action_format:
                    raise ValueError(
                        f"Mixed action formats detected: {inferred_action_format} vs {row_action_format} "
                        f"at line {line_idx + 1}"
                    )

                if self.action_dim <= 0:
                    self.action_dim = len(action_vec)

                if len(action_vec) != self.action_dim:
                    raise ValueError(
                        f"Expected action_dim={self.action_dim}, got {len(action_vec)} at line {line_idx + 1}"
                    )
                if len(state_vec) == 0:
                    raise ValueError(f"No state found at line {line_idx + 1}")

                episode_key = str(row.get("episode_key", row.get("episode_id", "episode_0")))
                frame_index = int(row.get("frame_index", line_idx))
                frames.append(RobotFrame(image_path, instruction, state_vec, action_vec, episode_key, frame_index))

        if not frames:
            raise ValueError(f"No frames found in {path}")
        if inferred_action_format is not None:
            self.action_format = inferred_action_format
        return frames

    def _build_sequence_indices(self, frames: List[RobotFrame], horizon: int) -> List[Tuple[int, List[int]]]:
        by_episode: Dict[str, List[int]] = {}
        for idx, fr in enumerate(frames):
            by_episode.setdefault(fr.episode_key, []).append(idx)
        samples: List[Tuple[int, List[int]]] = []
        for _, inds in by_episode.items():
            inds = sorted(inds, key=lambda i: frames[i].frame_index)
            for start_i in range(len(inds)):
                future = inds[start_i: start_i + horizon]
                if len(future) < horizon:
                    future = future + [future[-1]] * (horizon - len(future))
                samples.append((inds[start_i], future))
        return samples

    def _compute_action_stats(self) -> Dict[str, torch.Tensor]:
        actions = np.stack([self.frames[idx].action for idx, _ in self.samples], axis=0)
        mean = torch.tensor(actions.mean(axis=0), dtype=torch.float32)
        std = torch.tensor(actions.std(axis=0) + 1e-6, dtype=torch.float32)
        return {"mean": mean, "std": std}

    def _compute_state_stats(self) -> Dict[str, torch.Tensor]:
        states = np.stack([f.state for f in self.frames], axis=0)
        mean = torch.tensor(states.mean(axis=0), dtype=torch.float32)
        std = torch.tensor(states.std(axis=0) + 1e-6, dtype=torch.float32)
        return {"mean": mean, "std": std}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        x = x * 2.0 - 1.0
        return x

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cur_idx, future_indices = self.samples[idx]
        frame = self.frames[cur_idx]
        image = self._load_image(frame.image_path)

        text_tokens = self.text_processor.encode(frame.instruction)
        input_ids = text_tokens["input_ids"].long()
        attention_mask = text_tokens["attention_mask"].long()

        state = torch.tensor(frame.state, dtype=torch.float32)
        actions = torch.tensor([self.frames[j].action for j in future_indices], dtype=torch.float32)

        if self.normalize:
            state = (state - self.state_stats["mean"]) / self.state_stats["std"]
            actions = (actions - self.action_stats["mean"]) / self.action_stats["std"]

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state": state,
            "actions": actions,
            "instruction": frame.instruction,
        }


# ============================================================
# Model
# ============================================================

class VisionEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)
        x = self.proj(x)
        return self.norm(x)


class PretrainedVisionEncoder(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required for pretrained vision encoder")

        backbone_name = backbone_name.lower()
        if backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            feat_dim = 512
        elif backbone_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            feat_dim = 512
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported pretrained backbone: {backbone_name}")

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(feat_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).flatten(1)
        x = self.proj(x)
        return self.norm(x)


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    @staticmethod
    def sinusoidal(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        device = timesteps.device
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1))
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        x = self.sinusoidal(timesteps, self.dim)
        x = F.silu(self.fc1(x))
        return self.fc2(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class ConditionalResidualBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)
        self.cond_proj = nn.Linear(dim, dim * 2)
        self.time_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm1(x)
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        h = h + self.time_proj(t_emb).unsqueeze(1)
        x = x + self.dropout(self.attn(h))

        h2 = self.norm2(x)
        h2 = h2 * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        h2 = self.fc2(self.dropout(F.gelu(self.fc1(h2))))
        x = x + self.dropout(h2)
        return x


class ActionDiffusionPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 8, horizon: int = 4,
                 d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 vision_encoder_type: str = "simple", pretrained_backbone: str = "resnet18",
                 pretrained_vision: bool = True,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 freeze_clip_text: bool = False,
                 clip_pooling: str = "mean"):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.d_model = d_model
        self.vision_encoder_type = vision_encoder_type

        if vision_encoder_type == "simple":
            self.vision_encoder = VisionEncoder(out_dim=d_model)
        elif vision_encoder_type == "pretrained":
            self.vision_encoder = PretrainedVisionEncoder(
                backbone_name=pretrained_backbone,
                out_dim=d_model,
                pretrained=pretrained_vision,
            )
        else:
            raise ValueError(f"Unsupported vision_encoder_type: {vision_encoder_type}")

        self.text_encoder = CLIPTextEncoder(
            model_name=clip_model_name,
            out_dim=d_model,
            pretrained=True,
            freeze=freeze_clip_text,
            pooling=clip_pooling,
        )
        self.state_encoder = StateEncoder(state_dim=state_dim, out_dim=d_model)

        self.cond_fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.time_embed = TimestepEmbedding(d_model)
        self.action_in = nn.Linear(action_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=horizon + 8)
        self.blocks = nn.ModuleList([ConditionalResidualBlock(d_model, n_heads=n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, action_dim)

    def freeze_condition_encoders(self) -> None:
        freeze_module(self.vision_encoder)
        freeze_module(self.text_encoder)
        freeze_module(self.state_encoder)

    def unfreeze_diffusion_head(self) -> None:
        set_requires_grad(self.cond_fuse, True)
        set_requires_grad(self.time_embed, True)
        set_requires_grad(self.action_in, True)
        set_requires_grad(self.blocks, True)
        set_requires_grad(self.final_norm, True)
        set_requires_grad(self.out, True)

    def enable_diffusion_only_training(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.freeze_condition_encoders()
        self.unfreeze_diffusion_head()

    def condition(self, image: torch.Tensor, input_ids: torch.Tensor,
                  attention_mask: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        img_feat = self.vision_encoder(image)
        txt_feat = self.text_encoder(input_ids, attention_mask)
        st_feat = self.state_encoder(state)
        cond = torch.cat([img_feat, txt_feat, st_feat], dim=-1)
        return self.cond_fuse(cond)

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor,
                image: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        cond = self.condition(image, input_ids, attention_mask, state)
        t_emb = self.time_embed(timesteps)
        x = self.action_in(noisy_actions)
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x, cond, t_emb)
        x = self.final_norm(x)
        return self.out(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ============================================================
# Train / infer
# ============================================================

def save_checkpoint(path: str, model: ActionDiffusionPolicy, optimizer: torch.optim.Optimizer,
                    scheduler: DDPMScheduler, train_cfg: Dict,
                    dataset: OpenVLADeltaDataset, step: int) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "clip_text": {
            "model_name": train_cfg["clip_model_name"],
            "max_text_len": train_cfg["max_text_len"],
            "freeze_clip_text": train_cfg.get("freeze_clip_text", False),
            "clip_pooling": train_cfg.get("clip_pooling", "mean"),
        },
        "scheduler": {
            "betas": scheduler.betas.cpu(),
            "alphas": scheduler.alphas.cpu(),
            "alphas_cumprod": scheduler.alphas_cumprod.cpu(),
            "num_train_steps": scheduler.num_train_steps,
        },
        "train_cfg": train_cfg,
        "state_stats": None if dataset.state_stats is None else {k: v.cpu() for k, v in dataset.state_stats.items()},
        "action_stats": None if dataset.action_stats is None else {k: v.cpu() for k, v in dataset.action_stats.items()},
        "step": step,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: str = "cpu") -> Dict:
    return torch.load(path, map_location=device)


def build_optimizer_params(model: ActionDiffusionPolicy, args: argparse.Namespace):
    if args.use_lora:
        replaced = replace_linear_with_lora(
            model, attn_name_filter,
            r=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout
        )
        print(f"[INFO] LoRA replaced modules: {replaced}")

        if args.train_diffusion_only:
            model.enable_diffusion_only_training()
            for module_name, module in model.named_modules():
                if isinstance(module, LoRALinear):
                    module.A.requires_grad = False
                    module.B.requires_grad = False
                    if any(module_name.startswith(prefix) for prefix in ["cond_fuse", "time_embed", "action_in", "blocks", "out"]):
                        module.A.requires_grad = True
                        module.B.requires_grad = True
            optim_params = [p for p in model.parameters() if p.requires_grad]
        else:
            enable_lora_only(model)
            optim_params = list(lora_parameters(model))
    else:
        if args.train_diffusion_only:
            model.enable_diffusion_only_training()
        optim_params = [p for p in model.parameters() if p.requires_grad]

    if len(optim_params) == 0:
        raise RuntimeError("No trainable parameters found. Check train_diffusion_only / use_lora settings.")
    return optim_params


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    text_processor = CLIPTextProcessor(
        model_name=args.clip_model_name,
        max_length=args.max_text_len,
    )

    dataset = OpenVLADeltaDataset(
        jsonl_path=args.jsonl_path,
        image_root=args.image_root,
        text_processor=text_processor,
        image_size=args.image_size,
        horizon=args.horizon,
        action_dim=args.action_dim,
        normalize=not args.no_normalize,
        action_format=args.action_format,
    )
    args.action_dim = dataset.action_dim
    args.action_format = dataset.action_format
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    model = ActionDiffusionPolicy(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        horizon=args.horizon,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vision_encoder_type=args.vision_encoder_type,
        pretrained_backbone=args.pretrained_backbone,
        pretrained_vision=not args.no_pretrained_weights,
        clip_model_name=args.clip_model_name,
        freeze_clip_text=args.freeze_clip_text,
        clip_pooling=args.clip_pooling,
    ).to(device)

    if args.freeze_pretrained_vision and hasattr(model.vision_encoder, "backbone"):
        freeze_module(model.vision_encoder.backbone)

    optim_params = build_optimizer_params(model, args)

    total, trainable = count_parameters(model)
    print(f"[INFO] params total={total:,} trainable={trainable:,}")
    print(
        f"[INFO] dataset samples={len(dataset)} state_dim={dataset.state_dim} "
        f"action_dim={dataset.action_dim} action_format={dataset.action_format} horizon={args.horizon}"
    )
    print(f"[INFO] vision_encoder_type={args.vision_encoder_type} pretrained_backbone={args.pretrained_backbone}")
    print(f"[INFO] clip_model_name={args.clip_model_name} freeze_clip_text={args.freeze_clip_text} clip_pooling={args.clip_pooling}")
    print(f"[INFO] train_diffusion_only={args.train_diffusion_only}")

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    ddpm = DDPMScheduler(
        num_train_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    ).to(device)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        last_loss: Optional[float] = None
        for batch in loader:
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            state = batch["state"].to(device)
            actions = batch["actions"].to(device)

            noise = torch.randn_like(actions)
            timesteps = torch.randint(0, ddpm.num_train_steps, (actions.size(0),), device=device)
            noisy_actions = ddpm.add_noise(actions, noise, timesteps)

            pred_noise = model(noisy_actions, timesteps, image, input_ids, attention_mask, state)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(optim_params, args.grad_clip)
            optimizer.step()
            global_step += 1
            last_loss = float(loss.item())

            if global_step % args.log_every == 0:
                print(f"[train] epoch={epoch+1}/{args.epochs} step={global_step} loss={last_loss:.6f}")

        epoch_index = epoch + 1
        if args.save_every_epochs > 0 and epoch_index % args.save_every_epochs == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, f"policy_epoch_{epoch_index}.pt")
            train_cfg = vars(args).copy()
            train_cfg["last_epoch_loss"] = last_loss
            save_checkpoint(ckpt_path, model, optimizer, ddpm, train_cfg, dataset, global_step)
            if args.use_lora:
                torch.save(
                    state_dict_lora_only(model),
                    os.path.join(args.output_dir, f"policy_lora_epoch_{epoch_index}.pt")
                )
            loss_msg = "n/a" if last_loss is None else f"{last_loss:.6f}"
            print(f"[INFO] saved epoch checkpoint: {ckpt_path} (epoch={epoch_index}, last_loss={loss_msg})")

    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "policy_final.pt")
    save_checkpoint(final_path, model, optimizer, ddpm, vars(args).copy(), dataset, global_step)
    if args.use_lora:
        torch.save(
            state_dict_lora_only(model),
            os.path.join(args.output_dir, "policy_lora_final.pt")
        )
    print(f"[INFO] training complete. saved: {final_path}")


@torch.no_grad()
def sample_actions(model: ActionDiffusionPolicy, ddpm: DDPMScheduler,
                   image: torch.Tensor, input_ids: torch.Tensor,
                   attention_mask: torch.Tensor, state: torch.Tensor,
                   num_inference_steps: int, generator=None) -> torch.Tensor:
    device = image.device
    sample = torch.randn((image.size(0), model.horizon, model.action_dim), device=device, generator=generator)
    steps = np.linspace(ddpm.num_train_steps - 1, 0, num_inference_steps, dtype=np.int64)
    steps = list(dict.fromkeys(steps.tolist()))
    for t in steps:
        t_batch = torch.full((image.size(0),), int(t), device=device, dtype=torch.long)
        pred_noise = model(sample, t_batch, image, input_ids, attention_mask, state)
        sample = ddpm.step(pred_noise, int(t), sample, generator=generator)
    return sample


@torch.no_grad()
def predict_single(checkpoint_path: str, image_path: str, instruction: str, state: List[float],
                   image_size: Optional[int] = None, steps: int = 50, device: Optional[str] = None,
                   lora_path: Optional[str] = None) -> Dict:
    device = device or default_device()
    ckpt = load_checkpoint(checkpoint_path, device=device)
    cfg = ckpt["train_cfg"]

    text_processor = CLIPTextProcessor(
        model_name=cfg.get("clip_model_name", "openai/clip-vit-base-patch32"),
        max_length=cfg["max_text_len"],
    )

    model = ActionDiffusionPolicy(
        state_dim=len(state),
        action_dim=cfg["action_dim"],
        horizon=cfg["horizon"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        vision_encoder_type=cfg.get("vision_encoder_type", "simple"),
        pretrained_backbone=cfg.get("pretrained_backbone", "resnet18"),
        pretrained_vision=not cfg.get("no_pretrained_weights", False),
        clip_model_name=cfg.get("clip_model_name", "openai/clip-vit-base-patch32"),
        freeze_clip_text=cfg.get("freeze_clip_text", False),
        clip_pooling=cfg.get("clip_pooling", "mean"),
    ).to(device)

    if cfg.get("use_lora", False):
        _ = replace_linear_with_lora(
            model, attn_name_filter,
            r=cfg["lora_rank"], alpha=cfg["lora_alpha"], dropout=cfg["lora_dropout"]
        )
        model.load_state_dict(ckpt["model"])
        if lora_path is not None:
            load_state_dict_lora_only(model, torch.load(lora_path, map_location=device))
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    ddpm = DDPMScheduler(num_train_steps=ckpt["scheduler"]["num_train_steps"]).to(device)
    ddpm.register(
        ckpt["scheduler"]["betas"].to(device),
        ckpt["scheduler"]["alphas"].to(device),
        ckpt["scheduler"]["alphas_cumprod"].to(device),
    )

    size = image_size or cfg["image_size"]
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    image = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    image = image * 2.0 - 1.0

    text_tokens = text_processor.encode(instruction)
    input_ids = text_tokens["input_ids"].unsqueeze(0).to(device)
    attention_mask = text_tokens["attention_mask"].unsqueeze(0).to(device)

    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    state_stats = ckpt.get("state_stats")
    action_stats = ckpt.get("action_stats")
    if state_stats is not None:
        state_mean = state_stats["mean"].to(device)
        state_std = state_stats["std"].to(device)
        state_t = (state_t - state_mean) / state_std

    actions = sample_actions(
        model, ddpm, image, input_ids, attention_mask, state_t,
        num_inference_steps=steps
    )

    if action_stats is not None:
        action_mean = action_stats["mean"].to(device).view(1, 1, -1)
        action_std = action_stats["std"].to(device).view(1, 1, -1)
        actions = actions * action_std + action_mean

    actions_np = actions[0].cpu().numpy().tolist()
    first = actions_np[0]
    action_format = cfg.get("action_format", "cartesian_delta")
    return {
        "action_format": action_format,
        "predicted_action_sequence": actions_np,
        "first_action": format_output_action(first, action_format),
    }


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diffusion Policy VLA with CLIP text encoder + ResNet vision encoder + concat MLP fusion"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--jsonl_path", type=str, required=True)
    tr.add_argument("--image_root", type=str, required=True)
    tr.add_argument("--output_dir", type=str, required=True)
    tr.add_argument("--device", type=str, default=default_device())
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--batch_size", type=int, default=8)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--num_workers", type=int, default=0)
    tr.add_argument("--log_every", type=int, default=20)
    tr.add_argument("--save_every", type=int, default=500,
                    help="Deprecated step-based checkpoint interval. Kept for backward compatibility.")
    tr.add_argument("--save_every_epochs", type=int, default=100,
                    help="Save intermediate checkpoints every N epochs. Set 0 to disable.")
    tr.add_argument("--image_size", type=int, default=224)
    tr.add_argument("--max_text_len", type=int, default=48)
    tr.add_argument("--horizon", type=int, default=4)
    tr.add_argument("--action_dim", type=int, default=0)
    tr.add_argument("--action_format", type=str, default="auto", choices=["auto", "cartesian_delta", "joint_delta"])
    tr.add_argument("--d_model", type=int, default=256)
    tr.add_argument("--n_layers", type=int, default=6)
    tr.add_argument("--n_heads", type=int, default=8)
    tr.add_argument("--diffusion_steps", type=int, default=1000)
    tr.add_argument("--beta_start", type=float, default=0.00085)
    tr.add_argument("--beta_end", type=float, default=0.0120)
    tr.add_argument("--no_normalize", action="store_true")
    tr.add_argument("--use_lora", action="store_true")
    tr.add_argument("--lora_rank", type=int, default=8)
    tr.add_argument("--lora_alpha", type=int, default=8)
    tr.add_argument("--lora_dropout", type=float, default=0.0)

    tr.add_argument("--vision_encoder_type", type=str, default="pretrained", choices=["simple", "pretrained"])
    tr.add_argument("--pretrained_backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    tr.add_argument("--no_pretrained_weights", action="store_true")
    tr.add_argument("--freeze_pretrained_vision", action="store_true")

    tr.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    tr.add_argument("--freeze_clip_text", action="store_true")
    tr.add_argument("--clip_pooling", type=str, default="mean", choices=["mean", "eos"])

    tr.add_argument("--train_diffusion_only", action="store_true",
                    help="Freeze vision/text/state encoders and train only diffusion-side modules")

    pr = sub.add_parser("predict")
    pr.add_argument("--checkpoint", type=str, required=True)
    pr.add_argument("--image_path", type=str, required=True)
    pr.add_argument("--instruction", type=str, required=True)
    pr.add_argument("--state_json", type=str, required=True,
                    help='JSON string, e.g. "[joint7..., gripper, ee_pose7...]"')
    pr.add_argument("--steps", type=int, default=50)
    pr.add_argument("--device", type=str, default=default_device())
    pr.add_argument("--image_size", type=int, default=None)
    pr.add_argument("--lora_path", type=str, default=None)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        state = json.loads(args.state_json)
        out = predict_single(
            checkpoint_path=args.checkpoint,
            image_path=args.image_path,
            instruction=args.instruction,
            state=state,
            steps=args.steps,
            device=args.device,
            image_size=args.image_size,
            lora_path=args.lora_path,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))

## 실행
# python3 src/robotarm_project/diffusion_vla/diffusion_vla_pretrained.py train \
#   --jsonl_path src/robotarm_project/debug_runs/chair_grasp_openvla/merged_samples_openvla_joint_delta.jsonl \
#   --image_root src/robotarm_project/debug_runs/chair_grasp_openvla \
#   --output_dir src/robotarm_project/diffusion_vla/diffusion_policy_output_joint \
#   --device cuda \
#   --action_format joint_delta \
#   --action_dim 9


## Docker 에서 실행 예시
#     경로를 Docker 기준으로 바꿔 !
#     도커 안에서 pip install torchvision 설치 필수

# 기존 openvla-lora 이미지 활용하면서 새 컨테이너를 mount로 띄우기
# docker run --gpus all -it --name openvla_diffusion \
#   -v /home/lst7910/isaac_ros2_ws:/workspace \
#   -w /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/openvla \
#   openvla-lora:latest \
#   /bin/bash

# 기존 컨테이너에 들어가기
# docker start openvla_diffusion
# docker exec -it openvla_diffusion /bin/bash

# 그다음 도커 안에서 실행:
# python /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/diffusion_vla/diffusion_vla_pretrained.py train \
#   --jsonl_path /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/merged_samples_openvla_joint_delta_phase_aware.jsonl \
#   --image_root /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla \
#   --output_dir /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/diffusion_vla/diffusion_policy_output \
#   --device cuda \
#   --epochs 300 \
#   --save_every_epochs 100 \
#   --batch_size 2 \
#   --lr 5e-5 \
#   --vision_encoder_type pretrained \
#   --pretrained_backbone resnet18 \
#   --freeze_pretrained_vision \
#   --clip_model_name openai/clip-vit-base-patch32 \
#   --freeze_clip_text \
#   --clip_pooling mean \
#   --train_diffusion_only

## 추론 예시
# python /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/diffusion_vla/diffusion_vla_pretrained.py predict \
#   --checkpoint /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/diffusion_vla/diffusion_policy_output/policy_final.pt \
#   --image_path /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/episode_0005/images/sample_000000.jpg \
#   --instruction "의자를 집어" \
#   --state_json "[0.6289, 1.036, 0.3968, -0.6516, -0.34, 1.6468, -1.49, 0.08, 0.5622664110990461, 0.5539575737785446, 0.36995270014208675, 7.761320001389953e-06, 0.9999999993784533, -3.0746761244511756e-05, -1.541077514095348e-05]" \
#   --steps 50 \
#   --device cuda

# state_json은 episode_0005 폴더에 samples_openvla_delta.jsonl 에서 첫번째 이미지에 대응하는 state 값
