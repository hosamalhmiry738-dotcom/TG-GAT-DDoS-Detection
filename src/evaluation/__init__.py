"""
Evaluation module for TG-GAT DDoS Detection System.

This module contains evaluation infrastructure including:
- Model testing and benchmarking
- Explainable AI (XAI) implementations
- Performance analysis tools
"""

from .test import ModelTester
from .xai import XAIExplainer

__all__ = [
    "ModelTester",
    "XAIExplainer"
]
