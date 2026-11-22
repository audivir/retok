"""Pure-Python drop-in for the `retok` package: copy this file next to your code.

`import retok` then works without Rust or a wheel. Only `numpy` is needed. The functions,
signatures, results and errors equal those of the compiled package, but everything runs in Python
on one thread with the `re` module. Use the compiled package for speed.

Patterns are Python regexes, so unlike the Rust `regex` crate they may use look-around and
back-references, and a few escapes differ. Use patterns that both engines accept.
"""

from __future__ import annotations

import re
import sys
import types
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

__all__ = ["decode", "encode", "encode_with_lengths", "tokenize"]

ARRAY_NDIM = 2


def _compile(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValueError(f"regex parse error: {e}") from e


def tokenize(pattern: str, text: str) -> list[str]:
    """Splits text into all non-overlapping matches of regex pattern."""
    return [m.group() for m in _compile(pattern).finditer(text)]


def _enclose(
    ids: list[int], bos: int, eos: int, bos_after_first: bool, eos_before_last: bool
) -> list[int]:
    """Inserts BOS and EOS ids around `ids`, which must be long enough for the mode."""
    if len(ids) < 1 + int(bos_after_first) + int(eos_before_last):
        raise ValueError("Too few tokens")
    head = ids[:1] if bos_after_first else []
    rest = ids[len(head) :]
    tail = rest[-1:] if eos_before_last else []
    body = rest[: len(rest) - len(tail)]
    return [*head, bos, *body, eos, *tail]


def encode_with_lengths(
    texts: Sequence[str],
    min_len: int,
    pattern: str,
    token_indices: Mapping[str, int],
    pad: str,
    bos: str,
    eos: str,
    bos_after_first: bool = False,
    eos_before_last: bool = False,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Tokenizes, encloses with BOS and EOS, pads, and encodes texts.

    Rows are padded to the longer of `min_len` and the longest enclosed row.

    Args:
        texts: Input texts.
        min_len: Minimum padded length.
        pattern: Regex whose non-overlapping matches are the tokens.
        token_indices: Mapping from tokens to indices.
        pad: Padding token.
        bos: Start-of-sequence token.
        eos: End-of-sequence token.
        bos_after_first: Whether the first token stays in front of BOS.
        eos_before_last: Whether the last token stays behind EOS.

    Returns:
        Padded 2-D id array and unpadded row lengths.

    Raises:
        ValueError: If a text has too few tokens for the chosen enclosing mode.
        KeyError: If a token is missing from `token_indices`.
    """
    regex = _compile(pattern)
    index = dict(token_indices)
    bos_id, eos_id = index[bos], index[eos]
    pad_id = index.get(pad)
    rows = [
        _enclose(
            [index[m.group()] for m in regex.finditer(text)],
            bos_id,
            eos_id,
            bos_after_first,
            eos_before_last,
        )
        for text in texts
    ]
    width = max([len(r) for r in rows], default=0)
    width = max(width, min_len)
    if pad_id is None and any(len(r) < width for r in rows):
        raise KeyError(pad)
    ids = np.full((len(rows), width), 0 if pad_id is None else pad_id, dtype=np.int32)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = row
    return ids, np.array([len(r) for r in rows], dtype=np.int32)


def encode(
    texts: Sequence[str],
    min_len: int,
    pattern: str,
    token_indices: Mapping[str, int],
    pad: str,
    bos: str,
    eos: str,
    bos_after_first: bool = False,
    eos_before_last: bool = False,
) -> NDArray[np.int32]:
    """Encodes texts to a padded 2-D id array."""
    return encode_with_lengths(
        texts, min_len, pattern, token_indices, pad, bos, eos, bos_after_first, eos_before_last
    )[0]


def decode(
    inputs: NDArray[np.int32],
    decode_map: Mapping[int, str],
    special_tokens: set[str] | None = None,
) -> list[str]:
    """Decodes a 2-D id array to strings, dropping special tokens.

    Args:
        inputs: Encoded sequences.
        decode_map: Mapping from indices to tokens.
        special_tokens: Tokens omitted from the output.

    Returns:
        One decoded string per row.

    Raises:
        ValueError: If inputs is not 2-D.
        KeyError: If an index is missing from `decode_map`.
    """
    arr = np.ascontiguousarray(inputs, dtype=np.int32)
    if arr.ndim != ARRAY_NDIM:
        raise ValueError("inputs must be 2-D")
    if arr.shape[1] == 0:
        return []
    special = special_tokens or set()
    table = dict(decode_map)
    return [
        "".join(t for t in map(table.__getitem__, row) if t not in special) for row in arr.tolist()
    ]


def _register_submodule() -> None:
    """Registers `retok._retok` with the functions of the compiled module."""
    module = types.ModuleType("retok._retok")
    module.tokenize = tokenize  # type: ignore[attr-defined]
    sys.modules[module.__name__] = module
    setattr(sys.modules[__name__], "_retok", module)  # noqa: B010


if __name__ == "retok":
    _register_submodule()
