//! Token estimate without a tokenizer dependency (port of base.ts countTokens).
//!
//! A flat len/4 badly mis-estimates code (many short symbol tokens) and CJK
//! (≈1 token/char), so this approximates BPE behaviour: count word-ish runs,
//! standalone symbols, and CJK characters, then blend with a byte-rate floor.
//! Within ~10–15% of real tokenizers for mixed code + prose; backends that
//! report exact usage always override it.

pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let mut cjk = 0usize;
    let mut ascii_len = 0usize;
    let mut words = 0usize;
    let mut symbols = 0usize;
    let mut in_word = false;
    for ch in text.chars() {
        let c = ch as u32;
        if (0x3040..=0x30ff).contains(&c)
            || (0x3400..=0x9fff).contains(&c)
            || (0xac00..=0xd7af).contains(&c)
            || (0xf900..=0xfaff).contains(&c)
        {
            cjk += 1;
        }
        if !ch.is_ascii() {
            // Non-ASCII is dropped from the ASCII stats (as in the TS version);
            // it also ends a word-ish run.
            in_word = false;
            continue;
        }
        ascii_len += 1;
        if ch.is_ascii_alphanumeric() || ch == '_' {
            if !in_word {
                words += 1;
                in_word = true;
            }
        } else {
            in_word = false;
            if !ch.is_ascii_whitespace() {
                symbols += 1;
            }
        }
    }
    let structural = (words as f64 * 1.3 + symbols as f64).ceil() as usize;
    let byte_floor = ascii_len.div_ceil(4);
    structural.max(byte_floor) + cjk
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        assert_eq!(estimate_tokens(""), 0);
        assert!(estimate_tokens("hello world") >= 2);
        let code = "fn main() { println!(\"hi\"); }";
        assert!(estimate_tokens(code) > code.len() / 4);
        assert_eq!(estimate_tokens("日本語"), 3);
    }
}
