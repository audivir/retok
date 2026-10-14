"""Rust-accelerated regex tokenizer and encoder/decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from retok import _retok

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

__all__ = ["decode", "encode", "encode_with_lengths", "tokenize"]


def tokenize(pattern: str, text: str) -> list[str]:
    """Splits text into all non-overlapping matches of regex pattern."""
    return _retok.tokenize(pattern, text)


def encode_with_lengths(  # noqa: PLR0913,PLR0917
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
    ids, lengths, n, width = _retok.encode_with_lengths(
        list(texts),
        min_len,
        pattern,
        dict(token_indices),
        pad,
        bos,
        eos,
        bos_after_first,
        eos_before_last,
    )
    return (
        np.frombuffer(ids, dtype=np.int32).reshape(n, width),
        np.frombuffer(lengths, dtype=np.int32),
    )


def encode(  # noqa: PLR0913,PLR0917
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
    if arr.ndim != 2:  # noqa: PLR2004
        raise ValueError("inputs must be 2-D")
    return _retok.decode(arr.tobytes(), arr.shape[1], dict(decode_map), special_tokens)
