# retok

Regex tokenizer and token encoder/decoder written in Rust, exposed to Python via PyO3.

## Prerequisites

- Python 3.10 or newer
- Rust toolchain (edition 2024)
- `maturin`

## Installation

```shell
maturin develop
```

## Usage

The tokenization pattern is passed as argument, as any regex string. Text is enclosed with BOS and
EOS tokens and padded. With `bos_after_first` the first token stays in front of BOS, and with
`eos_before_last` the last token stays behind EOS.

```python
import retok

pattern = r"\d+|[A-Za-z]+|[^\s\w]"
retok.tokenize(pattern, "foo 12, bar!")  # ["foo", "12", ",", "bar", "!"]

vocab = ["<pad>", "<s>", "</s>", "foo", "bar", "12", ",", "!"]
token_indices = {t: i for i, t in enumerate(vocab)}
ids, lengths = retok.encode_with_lengths(
    ["foo 12, bar!"],
    min_len=10,
    pattern=pattern,
    token_indices=token_indices,
    pad="<pad>",
    bos="<s>",
    eos="</s>",
)
retok.decode(ids, dict(enumerate(vocab)), {"<pad>", "<s>", "</s>"})  # ["foo12,bar!"]
```

## License

MIT. See `LICENSE`.
