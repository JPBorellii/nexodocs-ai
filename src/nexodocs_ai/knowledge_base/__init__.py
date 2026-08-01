"""Deterministic generation and validation of the fictitious knowledge base."""

from .generator import check, generate
from .validator import validate_repository

__all__ = ["check", "generate", "validate_repository"]
