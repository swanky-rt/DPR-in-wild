"""Insert SQL cleaning helpers and replace _extract_sql_candidate in pipeline.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "src" / "sql_grounding" / "pipeline.py"


def main() -> None:
    lines = PIPE.read_text(encoding="utf-8").splitlines(keepends=True)
    i_extract = next(i for i, l in enumerate(lines) if l.startswith("def _extract_sql_candidate"))
    i_fallback = next(i for i, l in enumerate(lines) if l.startswith("def _fallback_sql_from_schema"))

    # Original think-block regex from current file (line after def _extract)
    orig_block = "".join(lines[i_extract:i_fallback])
    import re as _re

    m = _re.search(r're\.sub\((r"[^"]+")', orig_block)
    if not m:
        raise SystemExit("Could not find re.sub pattern in _extract_sql_candidate")
    think_raw = m.group(1)  # includes r"..."

    helpers = f'''
def _clean_qwen_output(text: str) -> str:
    """
    Strip Qwen / reasoning-model chatter before SQL extraction.
    """
    s = text or ""
    for pattern in (
        r"\\`\\`\\`[\\s\\S]*?\\`\\`\\`",
        r"<reasoning>[\\s\\S]*?</reasoning>",
    ):
        s = re.sub(pattern, "", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub({think_raw}, "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    return s


def _first_sql_select_or_cte_start(s: str) -> Optional[int]:
    """
    First real SQL start. English 'with ...' is not SQLite WITH; require
    WITH name AS ( for CTEs.
    """
    sel = re.search(r"(?is)\\bSELECT\\b", s)
    cte = re.search(
        r"(?is)\\bWITH\\s+(?:RECURSIVE\\s+)?[A-Za-z_][\\w]*\\s+AS\\s*\\(",
        s,
    )
    positions = [m.start() for m in (sel, cte) if m]
    return min(positions) if positions else None


def _trim_trailing_prose_after_sql(sql: str) -> str:
    t = sql.strip()
    for split_re in (
        r"\\n\\n+(?:First|Here|This |The |Note |However|But |So |Looking|Wait )",
        r"\\n\\n+[A-Za-z][^\\n]{20,}",
    ):
        parts = re.split(split_re, t, maxsplit=1)
        if len(parts) > 1 and len(parts[0]) > 20:
            t = parts[0].strip()
            break
    return t


def _is_valid_sql_start(sql: str) -> bool:
    t = (sql or "").lstrip()
    if re.match(r"^select\\b", t, flags=re.IGNORECASE):
        return True
    if re.match(r"^with\\b", t, flags=re.IGNORECASE):
        return bool(
            re.match(
                r"^with\\s+(?:recursive\\s+)?[A-Za-z_][\\w]*\\s+as\\s*\\(",
                t,
                flags=re.IGNORECASE,
            )
        )
    return False

'''

    new_fn = f'''def _extract_sql_candidate(text: str) -> str:
    """
    Normalize model output and extract the SQL statement.
    Handles cases where reasoning models prepend <think>...</think>.
    """
    s = _clean_qwen_output(text or "")
    s = _strip_code_fence(s)
    s = s.strip()

    start = _first_sql_select_or_cte_start(s)
    if start is not None:
        s = s[start:].strip()
    s = _trim_trailing_prose_after_sql(s)
    if ";" in s:
        first, rest = s.split(";", 1)
        if rest.strip() and not re.match(r"^\\s*(select|with)\\b", rest, flags=re.IGNORECASE):
            s = first.strip()

    return s.strip()

'''

    if "_clean_qwen_output" in "".join(lines):
        print("Already patched")
        return

    out = "".join(lines[:i_extract]) + helpers + new_fn + "".join(lines[i_fallback:])
    PIPE.write_text(out, encoding="utf-8")
    print("Patched:", PIPE)


if __name__ == "__main__":
    main()
