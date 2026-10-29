"""Word tokens for the chat corpus.

"you: nice to meet you, sam." -> ["you:", "nice", "to", "meet", "you", ",", "sam", ".", "<eol>"]
Punctuation marks are their own tokens; each line ends in <eol>. The vocabulary is every
token in the grammar plus the novel-test words (so they can be read, though no training
text contains them).
"""
from __future__ import annotations

import re

import corpus as C

EOL = "<eol>"
_TOK = re.compile(r"you:|bot:|[a-z']+|[?!.,]")


def tokenize(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        if not line:
            continue
        out.extend(_TOK.findall(line))
        out.append(EOL)
    return out


def detok(tokens: list[str]) -> str:
    s = ""
    for t in tokens:
        if t == EOL:
            break
        if t in "?!.," or not s:
            s += t
        else:
            s += " " + t
    return s


def _vocab() -> list[str]:
    tr, va, te = C.splits()
    words = set(tokenize(C.render(tr) + C.render(va) + C.render(te)))
    for _ in range(3):
        words |= set(tokenize(C.render(C.exchanges(5000, 99))))
    words |= set(tokenize(C.render(C.novel_exchanges(400))))
    return sorted(words)


VOCAB = _vocab()
INDEX = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)
EOL_ID = INDEX[EOL]
BOT_ID = INDEX["bot:"]


def encode(text: str) -> list[int]:
    return [INDEX[t] for t in tokenize(text)]


if __name__ == "__main__":
    tr, _, _ = C.splits()
    ids = encode(C.render(tr))
    print(f"vocab {V}, train tokens {len(ids)} ({len(ids) / len(tr):.1f} per exchange)")
    print(VOCAB)
