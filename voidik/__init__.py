"""Voidik-0.8B model package."""

from .config import Config, ModelConfig
from .model import VoidikForCausalLM

__all__ = ["Config", "ModelConfig", "VoidikForCausalLM"]
