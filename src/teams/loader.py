# src/teams/loader.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import random, re

__all__ = ["TeamProvider", "load_text", "list_files", "normalize_for_format", "count_mon_blocks"]

# -------- file io --------
def list_files(dirpath: Path) -> List[Path]:
    return [p for p in dirpath.iterdir() if p.is_file() and not p.name.startswith(".")]

def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_bytes().decode("utf-8", errors="ignore")

# -------- normalization --------
_MON_SPLIT_RE = re.compile(r"\n\s*\n", re.MULTILINE)

def _split_blocks(raw: str) -> List[str]:
    return [b.strip("\n") for b in _MON_SPLIT_RE.split(raw.strip()) if b.strip()]

def count_mon_blocks(raw: str) -> int:
    return len(_split_blocks(raw))

def _normalize_gen1(raw: str) -> str:
    """
    Gen1: inject 'Ability: No Ability' if missing, strip lines that are irrelevant
    (Nature/EVs/IVs), keep move lines, keep levels if present.
    """
    blocks = _split_blocks(raw)
    fixed = []
    for b in blocks:
        if not b.strip():
            continue
        lines = [ln.rstrip() for ln in b.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        # Insert ability after header if missing
        if not any(ln.lower().startswith("ability:") for ln in lines):
            lines.insert(1, "Ability: No Ability")
        # Remove lines that may confuse older teambuilder pipelines for Gen1
        lines = [ln for ln in lines if not re.match(r"^(EVs:|IVs:|Nature:)\b", ln, flags=re.I)]
        fixed.append("\n".join(lines))
    return "\n\n".join(fixed) + "\n"

def normalize_for_format(raw: str, fmt: str) -> str:
    f = (fmt or "").lower()
    if f.startswith("gen1"):
        return _normalize_gen1(raw)
    # Other gens: no-op for now; add per-gen rules here later.
    return raw if raw.endswith("\n") else raw + "\n"

# -------- provider --------
@dataclass
class TeamProvider:
    """
    If dir_*=None -> fixed team strings.
    If dir_* given -> sample with retries and sanitize.
    """
    fmt: str
    dir_learner: Optional[Path] = None
    dir_opponent: Optional[Path] = None
    fixed_learner: Optional[str] = None
    fixed_opponent: Optional[str] = None
    refresh: str = "episode"    # "episode" or "once"
    min_mons: int = 6
    max_sample_tries: int = 5

    def _pick_one(self, pool: List[Path]) -> str:
        p = random.choice(pool)
        return load_text(p), str(p)

    def _ensure_ok(self, txt: str) -> bool:
        return count_mon_blocks(txt) >= self.min_mons

    def prepare_once(self) -> None:
        """If refresh='once', choose fixed texts from dirs now."""
        if self.refresh != "once":
            return
        if self.dir_learner:
            files = list_files(self.dir_learner)
            if files:
                t, _ = self._pick_one(files)
                self.fixed_learner = normalize_for_format(t, self.fmt)
        if self.dir_opponent:
            files = list_files(self.dir_opponent)
            if files:
                t, _ = self._pick_one(files)
                self.fixed_opponent = normalize_for_format(t, self.fmt)

    def sample(self) -> Tuple[str, Optional[str], str, Optional[str]]:
        """
        Returns: (learner_team_text, learner_path_or_none, opponent_team_text, opponent_path_or_none)
        """
        # Choose source (dir vs fixed) and sample with retries
        def _sample_dir_or_fixed(dirp: Optional[Path], fixed_txt: Optional[str]):
            if dirp:
                files = list_files(dirp)
                last_txt, last_path = "", None
                for _ in range(self.max_sample_tries):
                    raw, path = self._pick_one(files)
                    norm = normalize_for_format(raw, self.fmt)
                    if self._ensure_ok(norm):
                        return norm, path
                    last_txt, last_path = norm, path
                return last_txt, last_path
            else:
                txt = normalize_for_format(fixed_txt or "", self.fmt)
                return txt, None

        # if refresh='once', use the prepared fixed values
        l_fixed = self.fixed_learner if self.refresh == "once" else None
        o_fixed = self.fixed_opponent if self.refresh == "once" else None

        l_txt, l_path = _sample_dir_or_fixed(self.dir_learner, l_fixed or self.fixed_learner)
        o_txt, o_path = _sample_dir_or_fixed(self.dir_opponent, o_fixed or self.fixed_opponent)
        return l_txt, l_path, o_txt, o_path
