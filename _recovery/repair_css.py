"""Repair FRAG-corrupted CSS: strip markers, drop incomplete rules and
non-CSS prose, keep every COMPLETE rule in original order.

We deliberately do NOT dedupe conflicting rules. The corruption spliced two
design revisions together; keeping both in file order lets the CSS cascade
resolve them exactly as the assembled sheet would (last definition wins),
which avoids guessing which revision was 'right'. The only things removed
are provably-invalid: the =====FRAG===== marker text, blocks left unclosed
at EOF, stray '}' with no matching '{', and bare prose that is not CSS.

Every removal is printed so the caller can confirm only junk was dropped.
"""
import re
import sys


def tokenize_rules(css: str):
    """Yield (kind, text) where kind in {rule, atrule_semi, JUNK:*}.

    Walks char by char tracking brace depth, skipping /* */ comments and
    string literals so braces inside them never move the depth.
    """
    i, n = 0, len(css)
    chunk_start = 0
    depth = 0
    in_comment = False
    in_str = ""  # the active quote char, or ""
    while i < n:
        c = css[i]
        two = css[i:i + 2]
        if in_comment:
            if two == "*/":
                i += 2; in_comment = False; continue
            i += 1; continue
        if in_str:
            if c == "\\":
                i += 2; continue
            if c == in_str:
                in_str = ""
            i += 1; continue
        if two == "/*":
            i += 2; in_comment = True; continue
        if c in "\"'":
            in_str = c; i += 1; continue
        if c == "{":
            depth += 1; i += 1; continue
        if c == "}":
            if depth == 0:
                # stray closer at top level -> junk, skip it, reset chunk
                pre = css[chunk_start:i].strip()
                if pre:
                    yield ("JUNK:stray-before-close", pre)
                i += 1; chunk_start = i; continue
            depth -= 1
            i += 1
            if depth == 0:
                yield ("rule", css[chunk_start:i].strip())
                chunk_start = i
            continue
        if c == ";" and depth == 0:
            # top-level at-rule like @import ...; or bare junk. Strip any
            # leading comment so "/* note */ @import ...;" is still kept.
            seg = css[chunk_start:i + 1].strip()
            probe = re.sub(r"^\s*(/\*.*?\*/\s*)*", "", seg, flags=re.S)
            if probe.startswith("@"):
                yield ("atrule_semi", seg)
            elif seg:
                yield ("JUNK:top-level-semi", seg)
            i += 1; chunk_start = i; continue
        i += 1
    tail = css[chunk_start:].strip()
    if tail:
        if depth > 0:
            yield ("JUNK:unclosed-at-eof", tail)
        else:
            yield ("JUNK:trailing-prose", tail)


def repair(path: str) -> None:
    raw = open(path, encoding="utf-8-sig").read()
    # remove the marker lines outright
    n_markers = len(re.findall(r"^\s*=====FRAG=====\s*$", raw, re.M))
    raw = re.sub(r"^\s*=====FRAG=====\s*$", "", raw, flags=re.M)

    kept, dropped = [], []
    for kind, text in tokenize_rules(raw):
        if kind.startswith("JUNK:"):
            dropped.append((kind, text))
        else:
            kept.append(text)

    out = "\n".join(kept) + "\n"
    ob, cb = out.count("{"), out.count("}")

    # Which selectors appear ONLY in dropped chunks (i.e. would be lost)?
    def selectors(text):
        return set(re.findall(r"([.#][A-Za-z0-9_-]+)", text))
    kept_sel = selectors(out)
    lost = set()
    for _k, text in dropped:
        lost |= selectors(text)
    truly_lost = sorted(s for s in lost if s not in kept_sel)

    print(f"\n=== {path} ===")
    print(f"  FRAG markers removed : {n_markers}")
    print(f"  complete rules kept  : {len(kept)}")
    print(f"  junk chunks dropped  : {len(dropped)}")
    for kind, text in dropped:
        one = " ".join(text.split())
        print(f"    - [{kind}] {one[:90]}{'...' if len(one) > 90 else ''}")
    print(f"  result braces        : {{ {ob}  }} {cb}  "
          f"{'BALANCED' if ob == cb else 'STILL MISMATCHED'}")
    if truly_lost:
        print(f"  SELECTORS LOST (only in dropped chunks): {', '.join(truly_lost)}")
    else:
        print("  no selector is lost (every dropped chunk's selectors survive elsewhere)")

    if ob == cb and "--write" in sys.argv:
        open(path, "w", encoding="utf-8", newline="\n").write(out)
        print("  WROTE repaired file.")
    elif ob != cb:
        print("  NOT written (still unbalanced) -- needs manual look.")
    else:
        print("  dry-run (pass --write to save).")


if __name__ == "__main__":
    targets = [a for a in sys.argv[1:] if not a.startswith("--")]
    for t in targets:
        repair(t)
