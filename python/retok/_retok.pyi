def tokenize(pattern: str, text: str) -> list[str]: ...
def encode_with_lengths(
    texts: list[str],
    min_len: int,
    pattern: str,
    token_indices: dict[str, int],
    pad: str,
    bos: str,
    eos: str,
    bos_after_first: bool = False,
    eos_before_last: bool = False,
) -> tuple[bytes, bytes, int, int]: ...
def decode(
    buffer: bytes,
    width: int,
    decode_map: dict[int, str],
    special_tokens: set[str] | None = None,
) -> list[str]: ...
