# Video walkthrough — talking outline

Target 4 minutes. Single take. Don't read this aloud — these are the points to
hit and the exact places to have on screen. Have `src/huffman.py` open and a
terminal in the repo root.

---

## 0:00 — What this is (20 s)

> "This is a lossless audio compressor built on Huffman coding. It takes a WAV
> file, compresses it, and decompresses back to a byte-identical file."

Show the terminal, run it once so they see it work:

```
python cli.py compress tones.wav out.ahuf
python cli.py decompress out.ahuf back.wav
md5 -q tones.wav back.wav
```

> "Same digest — that's the whole claim."

---

## 0:20 — The core, and how it's organised (60 s)

Show the file tree briefly, then open `src/huffman.py`.

> "The algorithm lives in `huffman.py`. It knows nothing about audio — it works
> on any symbols. `codec.py` is the audio part, `bitio.py` packs bits,
> `cli.py` is the interface. Dependencies only point one way."

Scroll to **`build_tree`, line 80.**

> "This is Huffman itself. Every symbol goes into a min-heap keyed by
> frequency. Then I repeatedly pull the two lightest nodes, merge them under a
> parent whose weight is the sum, and push it back. When one node is left,
> that's the root."

Point at the loop, **lines 97–102.**

> "That's the whole algorithm — six lines. Everything else in this file is
> about making it usable."

Scroll to **`canonical_codes`, line 161.**

> "Here's the first real deviation from the textbook. The textbook reads codes
> off the tree as root-to-leaf paths. I don't keep the tree at all — I keep
> only each leaf's *depth*, and reassign codes canonically from the depths:
> sort by length, hand out consecutive integers, shift left when the length
> increases. Same construction DEFLATE uses."

> "Two reasons. The decoder only needs the lengths, which are far cheaper to
> store than a tree. And it makes the codes a function of the lengths alone."

*(That last sentence sets up the tricky part — don't explain why yet.)*

---

## 1:20 — The tricky part (60 s)

**Pick ONE. I'd take option A — it's a better story and it connects to
everything else.**

### Option A: tie-breaking and why canonical coding fixes it

Show **`build_tree` line 94**, point at the `sequence` variable.

> "This was the hardest thing to get right, and my first version didn't have
> it. I pushed `Node` objects into the heap keyed by weight alone. When two
> weights tie, Python falls through to comparing the Nodes themselves, and
> raises `TypeError`."

> "Adding a sequence number fixes the crash. But the crash was hiding a worse
> problem I didn't see for a while: when weights tie, the *shape* of the tree
> isn't determined by the frequencies. A different tie-break gives a different
> tree, so different codes."

> "That matters because the encoder and decoder are separate programs. If they
> build even slightly different trees from the same data, every bit after the
> first difference is garbage — and it fails silently, not loudly."

> "Canonical coding is what actually solves it. Because the codes depend only
> on the *lengths*, and the lengths are what I write into the file, it doesn't
> matter which tie-break the encoder used. The decoder reads lengths and
> derives the same codes every time. The sequence number just makes it
> deterministic run-to-run; canonical assignment is what makes it *correct*."

### Option B: the code table format (`codec.py`, module docstring at the top)

Use this instead if you'd rather talk about the container. The two-level
encoding — gamma-coded symbol gaps, then the code lengths themselves Huffman
coded with a second small table — is genuinely fiddly, and it took the table
from 5 bits per entry down to about 1–2.

---

## 2:20 — The invariant (60 s)

State it plainly first:

> "The invariant is the prefix-free property: **no code is a prefix of any
> other code.**"

> "That's what makes variable-length codes decodable at all. If one code were a
> prefix of another, then when the decoder has read those bits it can't know
> whether to stop or keep going."

**Where it's established** — scroll back to `build_tree`, **line 100.**

> "It's established by construction. Every symbol is a *leaf*, and codes are
> root-to-leaf paths. You can't reach one leaf by passing through another, so
> no code can be a prefix of another. I get it for free from the tree shape."

**Where it's checked** — `Decoder.__init__`, **line 225.**

> "But the decoder doesn't get the tree — it gets lengths read out of a file,
> which might be corrupt. So here I check the Kraft inequality: the lengths
> only describe a valid prefix code if the sum of 2^(−length) over all symbols
> is at most 1. If it's over, the codes would overlap and I reject the file
> before decoding anything."

**Where it's relied on** — `decode_one`, **lines 264–269.**

> "And here's where it pays off. I read one bit at a time, accumulate, and
> check whether what I have is a valid code at this length. The moment it
> matches, I return — no lookahead, no backtracking. That's only safe *because*
> of the invariant. If a shorter code could be a prefix of a longer one, this
> loop would return the wrong symbol and never know."

---

## 3:20 — What breaks (45 s)

Use `build_tree`, **line 94** — the `sequence` in the heap key.

> "If I delete the sequence number from this key, two things go wrong."

Actually do it — comment it out and run the tests on camera:

```
python -m pytest tests/ -q
```

> "First: `TypeError`, because with tied weights Python tries to compare two
> `Node` objects and there's no ordering defined for them. That's the loud
> failure, and it's the easy one."

> "The second is the interesting one. Suppose I'd made `Node` comparable
> instead — say, by memory address. The crash goes away, the tests probably
> pass, and the tree shape now depends on where Python happened to allocate
> objects. Two runs on the same input could produce different trees."

> "For this codebase that's survivable, because canonical coding means I only
> ever write lengths. But if I'd taken the obvious route and serialised the
> tree instead, a file compressed on one run could fail to decode on the next,
> intermittently, with no error — just wrong audio."

Undo the change, re-run, show green.

---

## Closing (15 s)

> "So: Huffman in `huffman.py`, prediction and block structure in `codec.py`,
> and the thing holding it together is that the decoder only ever needs code
> *lengths*, never the tree."

---

## Checklist before recording

- [ ] `tones.wav` (or a real track) in the repo root so the demo runs
- [ ] Terminal font large enough to read
- [ ] Editor at ~14pt+, line numbers on
- [ ] Tests currently green
- [ ] `git stash` any uncommitted mess so the diff is clean on camera
- [ ] Do the "what breaks" edit *live* — it's much more convincing than
      describing it

## Timing

| section | target | running |
|---|---|---|
| what it is | 0:20 | 0:20 |
| core + organisation | 1:00 | 1:20 |
| tricky part | 1:00 | 2:20 |
| invariant | 1:00 | 3:20 |
| what breaks | 0:45 | 4:05 |
| close | 0:15 | 4:20 |

Under the 5-minute cap with room to breathe. If you overrun, cut the closing
and shorten the opening demo — the four required elements are what's marked.
