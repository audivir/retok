"""Generic tests for the tokenizer, encoder, and decoder."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pytest
import retok


class Kwargs(TypedDict):
    pattern: str
    token_indices: dict[str, int]
    pad: str
    bos: str
    eos: str


VOCAB = ["<pad>", "<s>", "</s>", "a", "b", "ab", "c"]
INDEX = {t: i for i, t in enumerate(VOCAB)}
KW: Kwargs = {
    "pattern": r"ab|\w",
    "token_indices": INDEX,
    "pad": "<pad>",
    "bos": "<s>",
    "eos": "</s>",
}


@pytest.mark.parametrize(
    ("pattern", "text", "expected"),
    [
        (r"ab|\w", "abcb", ["ab", "c", "b"]),
        (r"\w|ab", "abcb", ["a", "b", "c", "b"]),
        (r"\d+|[A-Za-z]+|[^\s\w]", "foo 12, bar!", ["foo", "12", ",", "bar", "!"]),
        (r"\w+", "naïve café 日本語", ["naïve", "café", "日本語"]),
        (r"\d", "no digits", []),
        (r"\w+", "", []),
    ],
)
def test_tokenize(pattern: str, text: str, expected: list[str]) -> None:
    assert retok.tokenize(pattern, text) == expected


def test_tokenize_invalid_regex() -> None:
    with pytest.raises(ValueError, match="regex"):
        retok.tokenize("(", "x")


@pytest.mark.parametrize(
    ("after_first", "before_last", "expected"),
    [
        (False, False, [1, 3, 5, 6, 2, 0, 0]),
        (False, True, [1, 3, 5, 2, 6, 0, 0]),
        (True, False, [3, 1, 5, 6, 2, 0, 0]),
        (True, True, [3, 1, 5, 2, 6, 0, 0]),
    ],
)
def test_enclose_modes_and_padding(
    after_first: bool, before_last: bool, expected: list[int]
) -> None:
    ids, lengths = retok.encode_with_lengths(
        ["aabc"], 7, bos_after_first=after_first, eos_before_last=before_last, **KW
    )
    assert ids.tolist() == [expected]
    assert lengths.tolist() == [5]


def test_rows_grow_beyond_min_len() -> None:
    ids, lengths = retok.encode_with_lengths(["a", "abccc"], 2, **KW)
    assert ids.shape == (2, 6)
    assert lengths.tolist() == [3, 6]


def test_decode_roundtrip_drops_special_tokens() -> None:
    ids = retok.encode(["abca", "c"], 4, **KW)
    decoded = retok.decode(ids, dict(enumerate(VOCAB)), {"<pad>", "<s>", "</s>"})
    assert decoded == ["abca", "c"]


def test_errors() -> None:
    updated_kwargs: Kwargs = {**KW, "pattern": r"\w"}
    with pytest.raises(KeyError):
        retok.encode(["z"], 5, **updated_kwargs)
    with pytest.raises(ValueError, match="Too few"):
        retok.encode([""], 5, **KW)
    with pytest.raises(ValueError, match="Too few"):
        retok.encode(["a"], 5, bos_after_first=True, eos_before_last=True, **KW)


def test_decode_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        retok.decode(np.array([1, 2, 3], dtype=np.int32), dict(enumerate(VOCAB)))
