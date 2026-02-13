from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from collections import Counter


@dataclass(frozen=True)
class VocabConfig:
    min_freq: int = 2
    max_size: int = 20000
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"


@dataclass(frozen=True)
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    cfg: VocabConfig

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.cfg.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.cfg.unk_token]

    def encode_token(self, tok: str) -> int:
        return self.token_to_id.get(tok, self.unk_id)

    def decode_id(self, i: int) -> str:
        if 0 <= i < len(self.id_to_token):
            return self.id_to_token[i]
        return self.cfg.unk_token

    def to_json(self) -> Dict:
        return {
            "cfg": asdict(self.cfg),
            "token_to_id": self.token_to_id,
        }

    @staticmethod
    def from_json(d: Dict) -> "Vocab":
        cfg = VocabConfig(**d["cfg"])
        token_to_id = {str(k): int(v) for k, v in d["token_to_id"].items()}
        id_to_token = [""] * (max(token_to_id.values()) + 1)
        for t, i in token_to_id.items():
            id_to_token[i] = t
        return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, cfg=cfg)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "Vocab":
        return Vocab.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def build_vocab(token_sequences: Iterable[List[str]], cfg: VocabConfig) -> Tuple[Vocab, Counter]:
    """
    Build vocab from an iterable of token lists (typically TRAIN only).
    Returns vocab and raw Counter for reporting.
    """
    c: Counter = Counter()
    for toks in token_sequences:
        c.update(toks)

    # special tokens first
    id_to_token: List[str] = [cfg.pad_token, cfg.unk_token]
    token_to_id: Dict[str, int] = {cfg.pad_token: 0, cfg.unk_token: 1}

    # sort by freq desc, then lexicographically for determinism
    items = [(t, n) for t, n in c.items() if n >= cfg.min_freq and t not in token_to_id]
    items.sort(key=lambda x: (-x[1], x[0]))

    for t, _n in items[: max(0, cfg.max_size - len(id_to_token))]:
        token_to_id[t] = len(id_to_token)
        id_to_token.append(t)

    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, cfg=cfg), c
