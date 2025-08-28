# src/scheduling/tournament.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable

@dataclass(frozen=True)
class MatchSpec:
    learner_key: str     # "ac", etc. (registry key)
    opponent_key: str    # "random" now; later other agents
    # you can also put team provider ids here if needed

def single_match() -> Iterable[MatchSpec]:
    while True:
        yield MatchSpec(learner_key="ac", opponent_key="random")

def round_robin(agent_keys: List[str]) -> List[Tuple[str, str]]:
    """All-pairs (ordered) without self-matches."""
    return [(a, b) for a in agent_keys for b in agent_keys if a != b]
