"""
Models module for TG-GAT DDoS Detection System.

This module contains the core TG-GAT model architecture and custom layers
for DDoS attack detection using graph neural networks and transformers.
"""

from .tg_gat import TGGATModel

__all__ = [
    "TGGATModel"
]
