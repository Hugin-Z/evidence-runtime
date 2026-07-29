"""Exact Locator 的模块常量。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

from ..documents import NormalizedDocument
from ..rg_executor import (
    InvalidRipgrepOutputError,
    RipgrepExecutor,
    RipgrepLineEvent,
    RipgrepSubmatch,
)
from ..types import EvidenceContract, EvidenceSlot, EvidenceSpan, VectorCandidate
from ..vector_discovery import build_slot_query


CANONICAL_MANIFEST_NAME = "canonical-corpus-manifest.json"
MAX_ANCHORS = 24
MAX_ANCHOR_CHARS = 80
_MAX_RESCUE_CANDIDATES_PER_SLOT = 8
_ANCHOR_SOURCES = frozenset(
    {"caller", "slot_name", "required_field", "query", "candidate"}
)
