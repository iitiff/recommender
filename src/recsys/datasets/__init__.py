"""Public dataset adapters: raw files → the engine's standard layout
(catalog/ + interactions.csv [+ availability.csv]).

Each adapter documents its source and license. Raw and prepared data stay out
of git; see docs/DATASETS.md.
"""

from recsys.datasets import retailrocket

ADAPTERS = {retailrocket.NAME: retailrocket}

__all__ = ["ADAPTERS"]
