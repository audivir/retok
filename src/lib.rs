//! Regex tokenizer and token encoder/decoder.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use regex::Regex;

/// Failure conditions while encoding or decoding.
#[derive(Debug)]
enum EncodeError {
    TooFewTokens,
    UnknownToken(String),
    UnknownIndex(i32),
}

impl From<EncodeError> for PyErr {
    fn from(e: EncodeError) -> Self {
        match e {
            EncodeError::TooFewTokens => PyValueError::new_err("Too few tokens"),
            EncodeError::UnknownToken(t) => PyKeyError::new_err(t),
            EncodeError::UnknownIndex(i) => PyKeyError::new_err(i),
        }
    }
}

/// Returns all non-overlapping matches of `re` in `text`.
fn tokenize_with<'a>(re: &Regex, text: &'a str) -> Vec<&'a str> {
    re.find_iter(text).map(|m| m.as_str()).collect()
}

fn compile(pattern: &str) -> PyResult<Regex> {
    Regex::new(pattern).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Inserts BOS and EOS ids around `ids`.
///
/// With `bos_after_first`, the first id stays in front of BOS. With `eos_before_last`, the last id
/// stays behind EOS.
///
/// # Errors
///
/// Returns `Err` if `ids` is too short for the chosen mode.
fn enclose(
    ids: &[i32],
    bos: i32,
    eos: i32,
    bos_after_first: bool,
    eos_before_last: bool,
) -> Result<Vec<i32>, EncodeError> {
    if ids.len() < 1 + usize::from(bos_after_first) + usize::from(eos_before_last) {
        return Err(EncodeError::TooFewTokens);
    }
    let (head, rest) = if bos_after_first { ids.split_at(1) } else { ids.split_at(0) };
    let (body, tail) =
        if eos_before_last { rest.split_at(rest.len() - 1) } else { (rest, &[][..]) };
    let mut out = Vec::with_capacity(ids.len() + 2);
    out.extend_from_slice(head);
    out.push(bos);
    out.extend_from_slice(body);
    out.push(eos);
    out.extend_from_slice(tail);
    Ok(out)
}

/// Pads rows to the longer of `min_len` and the longest row, then flattens row-major.
///
/// Returns `(flat, lengths, width, needs_pad)`.
fn pad_rows(rows: &[Vec<i32>], pad: Option<i32>, min_len: usize) -> (Vec<i32>, Vec<i32>, usize, bool) {
    let lengths: Vec<i32> = rows.iter().map(|r| r.len() as i32).collect();
    let width = rows.iter().map(Vec::len).max().unwrap_or(0).max(min_len);
    let needs_pad = rows.iter().any(|r| r.len() < width);
    let fill = pad.unwrap_or(0);
    let mut flat = Vec::with_capacity(rows.len() * width);
    for r in rows {
        flat.extend_from_slice(r);
        flat.resize(flat.len() + (width - r.len()), fill);
    }
    (flat, lengths, width, needs_pad)
}

/// Decodes rows of ids, skipping `special` tokens.
///
/// # Errors
///
/// Returns `Err` if an id is missing from `decode_map`.
fn decode_rows(
    flat: &[i32],
    width: usize,
    decode_map: &HashMap<i32, String>,
    special: &HashSet<String>,
) -> Result<Vec<String>, EncodeError> {
    if width == 0 {
        return Ok(Vec::new());
    }
    flat.par_chunks(width)
        .map(|row| {
            let mut s = String::new();
            for id in row {
                let tok = decode_map.get(id).ok_or(EncodeError::UnknownIndex(*id))?;
                if !special.contains(tok) {
                    s.push_str(tok);
                }
            }
            Ok(s)
        })
        .collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// Splits `text` into all non-overlapping matches of regex `pattern`.
#[pyfunction]
fn tokenize(pattern: &str, text: &str) -> PyResult<Vec<String>> {
    let re = compile(pattern)?;
    Ok(tokenize_with(&re, text).into_iter().map(str::to_owned).collect())
}

/// Encodes texts to padded ids.
///
/// Returns `(ids, lengths, n_rows, width)` where both buffers are native-endian int32.
#[pyfunction]
#[pyo3(signature = (texts, min_len, pattern, token_indices, pad, bos, eos, bos_after_first=false, eos_before_last=false))]
#[allow(clippy::too_many_arguments)]
fn encode_with_lengths<'py>(
    py: Python<'py>,
    texts: Vec<String>,
    min_len: usize,
    pattern: &str,
    token_indices: HashMap<String, i32>,
    pad: &str,
    bos: &str,
    eos: &str,
    bos_after_first: bool,
    eos_before_last: bool,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize, usize)> {
    let re = compile(pattern)?;
    let get = |t: &str| {
        token_indices.get(t).copied().ok_or_else(|| PyKeyError::new_err(t.to_owned()))
    };
    let (bos_id, eos_id) = (get(bos)?, get(eos)?);
    let pad_id = token_indices.get(pad).copied();
    let n = texts.len();
    let (flat, lengths, width, needs_pad) = py.detach(|| -> Result<_, EncodeError> {
        let rows = texts
            .par_iter()
            .map(|text| {
                let ids = tokenize_with(&re, text)
                    .into_iter()
                    .map(|t| {
                        token_indices
                            .get(t)
                            .copied()
                            .ok_or_else(|| EncodeError::UnknownToken(t.to_owned()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                enclose(&ids, bos_id, eos_id, bos_after_first, eos_before_last)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(pad_rows(&rows, pad_id, min_len))
    })?;
    if needs_pad && pad_id.is_none() {
        return Err(PyKeyError::new_err(pad.to_owned()));
    }
    Ok((PyBytes::new(py, &i32_bytes(&flat)), PyBytes::new(py, &i32_bytes(&lengths)), n, width))
}

/// Decodes a row-major native-endian int32 buffer with rows of `width` ids.
#[pyfunction]
#[pyo3(signature = (buffer, width, decode_map, special_tokens=None))]
fn decode(
    py: Python<'_>,
    buffer: &[u8],
    width: usize,
    decode_map: HashMap<i32, String>,
    special_tokens: Option<HashSet<String>>,
) -> PyResult<Vec<String>> {
    let (chunks, remainder) = buffer.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(PyValueError::new_err("buffer length must be a multiple of 4"));
    }
    let flat: Vec<i32> = chunks.iter().map(|c| i32::from_ne_bytes(*c)).collect();
    let special = special_tokens.unwrap_or_default();
    Ok(py.detach(|| decode_rows(&flat, width, &decode_map, &special))?)
}

#[pymodule]
fn _retok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(encode_with_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enclose_modes() {
        let v = [1, 2, 3];
        assert_eq!(enclose(&v, 8, 9, false, false).unwrap(), [8, 1, 2, 3, 9]);
        assert_eq!(enclose(&v, 8, 9, false, true).unwrap(), [8, 1, 2, 9, 3]);
        assert_eq!(enclose(&v, 8, 9, true, false).unwrap(), [1, 8, 2, 3, 9]);
        assert_eq!(enclose(&v, 8, 9, true, true).unwrap(), [1, 8, 2, 9, 3]);
        assert!(enclose(&[1, 2], 8, 9, true, true).is_err());
        assert!(enclose(&[], 8, 9, false, false).is_err());
    }

    #[test]
    fn tokenize_words_and_numbers() {
        let re = Regex::new(r"\d+|[A-Za-z]+|[^\s\w]").unwrap();
        assert_eq!(tokenize_with(&re, "foo 12, bar!"), ["foo", "12", ",", "bar", "!"]);
    }
}
