//! Windows PowerShell 5.1 (what `run_shell` / `shell_session` spawn on Windows)
//! has no `&&` / `||` pipeline-chain operators — they arrived in PowerShell 7.
//! Small models write bash-style chains constantly ("python x.py && echo ok"),
//! and 5.1 answers with "The token '&&' is not a valid statement separator",
//! which then sends the model into a retry loop. Rewrite unquoted chains into
//! the equivalent `$?` checks so the command just works:
//!
//! ```text
//! a && b        →  a; if ($?) { b }
//! a || b        →  a; if (-not $?) { b }
//! a && b || c   →  a; if ($?) { b }; if (-not $?) { c }
//! ```
//!
//! Quoted text and `&&`/`||` inside quotes are left untouched. Commands without
//! an unquoted chain operator are returned unchanged.

#[derive(Clone, Copy, PartialEq, Eq)]
enum ChainOp {
    And,
    Or,
}

/// Rewrite bash-style `&&` / `||` chains for Windows PowerShell 5.1.
pub fn adapt_chains_for_powershell(cmd: &str) -> String {
    let mut parts: Vec<(Option<ChainOp>, String)> = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut pending: Option<ChainOp> = None;
    let chars: Vec<char> = cmd.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if let Some(q) = quote {
            cur.push(ch);
            if ch == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        if ch == '"' || ch == '\'' {
            quote = Some(ch);
            cur.push(ch);
            i += 1;
            continue;
        }
        let next = chars.get(i + 1).copied();
        let op = match (ch, next) {
            ('&', Some('&')) => Some(ChainOp::And),
            ('|', Some('|')) => Some(ChainOp::Or),
            _ => None,
        };
        if let Some(op) = op {
            parts.push((pending, cur.trim().to_string()));
            pending = Some(op);
            cur.clear();
            i += 2;
            continue;
        }
        cur.push(ch);
        i += 1;
    }
    parts.push((pending, cur.trim().to_string()));
    if parts.len() == 1 {
        return cmd.to_string(); // no unquoted chain → untouched
    }
    let out: Vec<String> = parts
        .into_iter()
        .filter(|(_, text)| !text.is_empty())
        .map(|(op, text)| match op {
            Some(ChainOp::And) => format!("if ($?) {{ {text} }}"),
            Some(ChainOp::Or) => format!("if (-not $?) {{ {text} }}"),
            None => text,
        })
        .collect();
    out.join("; ")
}

/// True when this process will run commands through Windows PowerShell 5.1.
pub fn uses_windows_powershell() -> bool {
    cfg!(windows)
}

#[cfg(test)]
mod tests {
    use super::*;

    // bugfixes2026_09: PowerShell 5.1 chain compatibility
    #[test]
    fn rewrites_unquoted_chains() {
        assert_eq!(adapt_chains_for_powershell("python x.py && echo ok"), "python x.py; if ($?) { echo ok }");
        assert_eq!(adapt_chains_for_powershell("npm test || echo failed"), "npm test; if (-not $?) { echo failed }");
        assert_eq!(adapt_chains_for_powershell("a && b || c"), "a; if ($?) { b }; if (-not $?) { c }");
    }

    #[test]
    fn leaves_quoted_operators_and_plain_commands_alone() {
        assert_eq!(adapt_chains_for_powershell("echo \"a && b\""), "echo \"a && b\"");
        assert_eq!(adapt_chains_for_powershell("Get-Process | Select -First 3"), "Get-Process | Select -First 3");
        assert_eq!(
            adapt_chains_for_powershell("python -c \"print('x' or 'y')\""),
            "python -c \"print('x' or 'y')\""
        );
    }
}
