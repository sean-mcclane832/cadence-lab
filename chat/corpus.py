"""A small, self-written chat corpus.

Every exchange is two lines:

    you: <prompt>
    bot: <reply>

The generator is a list of intents. Each intent draws a prompt and returns the prompt with
the full set of replies that count as a sensible answer to it, so a generated reply can be
scored as valid or not (``valid_reply``), on top of next-character accuracy.

Some replies need something from the prompt (the name in "my name is sam" or the thing in
"i like pizza"). Those test whether the model carries context, not just a frequency table.

Everything here is written for this lab; no outside text.
"""
from __future__ import annotations

import random

ALPHABET = "abcdefghijklmnopqrstuvwxyz ?!.,'\n:"
INDEX = {c: i for i, c in enumerate(ALPHABET)}
V = len(ALPHABET)

NAMES = ["sam", "alex", "maria", "tom", "nina", "leo", "kate", "max", "lucy", "ben",
         "emma", "jack", "rosa", "dan", "ivy", "omar"]
FOODS = ["pizza", "pasta", "rice", "soup", "tacos", "cake", "apples", "bread", "eggs", "fish"]
COLORS = ["red", "blue", "green", "black", "white", "yellow", "purple", "orange"]
ANIMALS = ["dog", "cat", "bird", "fish", "horse", "rabbit", "snake", "frog"]
HOBBIES = ["music", "games", "books", "movies", "running", "football", "chess", "art"]
PLACES = ["spain", "rome", "paris", "london", "the beach", "the park", "the city", "home"]
MOODS_GOOD = ["good", "great", "fine", "happy", "okay"]
MOODS_BAD = ["sad", "tired", "bored", "sick", "angry"]

# Things the bot "likes" (fixed persona), so yes/no answers are determined by the prompt.
BOT_LIKES = {"pizza", "cake", "soup", "bread", "music", "books", "chess", "art", "dog",
             "cat", "bird", "blue", "green", "rome", "the beach"}


def _likes(x: str) -> bool:
    return x in BOT_LIKES


# Fillers never seen in training, for the copy test (novel_exchanges).
NOVEL = {"names": ["zoe", "hugo", "pablo", "wendy"], "things": ["noodles", "goats", "berlin", "kites"]}


def intents(r: random.Random, k: int | None = None) -> tuple[str, list[str], str]:
    """One exchange: (prompt, valid replies, intent name)."""
    k = r.randrange(16) if k is None else k
    if k == 0:
        g = r.choice(["hi", "hello", "hey", "hi there", "hello there", "good morning"])
        return g, ["hi!", "hello!", "hey, how are you?", "hi, how are you?", "hello, nice to see you."], "greet"
    if k == 1:
        p = r.choice(["how are you?", "how are you doing?", "how is it going?", "are you okay?"])
        return p, ["i am good, thanks. and you?", "i am fine, thank you.", "pretty good, how about you?"], "howareyou"
    if k == 2:
        m = r.choice(MOODS_GOOD)
        p = r.choice([f"i am {m}", f"i am {m}, thanks", f"i feel {m}"])
        return p, ["that is good to hear.", "nice, i am glad.", "great!"], "mood_good"
    if k == 3:
        m = r.choice(MOODS_BAD)
        p = r.choice([f"i am {m}", f"i feel {m}", f"i am a bit {m}"])
        return p, ["i am sorry to hear that.", "oh no, what happened?", "that is too bad."], "mood_bad"
    if k == 4:
        n = r.choice(NAMES)
        p = r.choice([f"my name is {n}", f"i am {n}", f"call me {n}"])
        return p, [f"nice to meet you, {n}.", f"hi {n}!", f"hello {n}, nice to meet you."], "name"
    if k == 5:
        p = r.choice(["what is your name?", "who are you?", "what are you?"])
        return p, ["i am cadence, a small brain.", "my name is cadence.", "i am a bot called cadence."], "botname"
    if k == 6:
        f = r.choice(FOODS + HOBBIES + ANIMALS + COLORS + PLACES)
        p = f"do you like {f}?"
        if _likes(f):
            return p, [f"yes, i like {f}.", f"yes, i love {f}!", "yes, a lot."], "doyoulike"
        return p, [f"no, i do not like {f}.", f"not really.", f"no, not {f}."], "doyoulike"
    if k == 7:
        f = r.choice(FOODS + HOBBIES + ANIMALS)
        p = r.choice([f"i like {f}", f"i love {f}", f"i really like {f}"])
        return p, [f"{f} is nice.", f"cool, i like {f} too." if _likes(f) else f"oh, i do not like {f}.",
                   f"why do you like {f}?"], "ilike"
    if k == 8:
        p = r.choice(["what is your favorite color?", "what color do you like?"])
        return p, ["i like blue.", "blue, and green too.", "my favorite color is blue."], "color"
    if k == 9:
        p = r.choice(["what is your favorite food?", "what do you like to eat?"])
        return p, ["i like pizza.", "pizza, and cake.", "my favorite food is pizza."], "food"
    if k == 10:
        p = r.choice(["thanks", "thank you", "thanks a lot", "thank you so much"])
        return p, ["you are welcome.", "no problem.", "any time!"], "thanks"
    if k == 11:
        p = r.choice(["bye", "goodbye", "see you", "see you later", "good night"])
        return p, ["bye!", "see you soon.", "goodbye, take care.", "good night!"], "bye"
    if k == 12:
        a = r.choice(ANIMALS)
        p = r.choice([f"i have a {a}", f"i got a {a}", f"my pet is a {a}"])
        return p, [f"a {a}? that is cool.", f"what is your {a} called?", f"i like {a}s."], "pet"
    if k == 13:
        pl = r.choice(PLACES)
        p = r.choice([f"i am going to {pl}", f"i went to {pl}", f"i want to go to {pl}"])
        return p, [f"{pl} sounds fun.", f"have fun in {pl}!", f"what will you do in {pl}?"], "place"
    if k == 14:
        p = r.choice(["what can you do?", "can you help me?", "what do you do?"])
        return p, ["i can talk a little.", "i can try to help.", "not much, i am still learning."], "cando"
    p = r.choice(["are you a robot?", "are you real?", "are you alive?", "are you smart?"])
    return p, ["i am a small brain made of numbers.", "not really, i am still learning.", "i am a bot."], "robot"


def exchanges(n: int, seed: int) -> list[tuple[str, str, list[str], str]]:
    """n exchanges: (prompt, the reply shown, all valid replies, intent)."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        p, valid, name = intents(r)
        out.append((p, r.choice(valid), valid, name))
    return out


def novel_exchanges(n: int, seed: int = 4) -> list[tuple[str, str, list[str], str]]:
    """Exchanges whose reply should copy a word never seen in training (a name or a liked
    thing). Valid replies are built the same way as in training."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        if i % 2 == 0:
            w = r.choice(NOVEL["names"])
            p = r.choice([f"my name is {w}", f"i am {w}", f"call me {w}"])
            valid = [f"nice to meet you, {w}.", f"hi {w}!", f"hello {w}, nice to meet you."]
            name = "novel_name"
        else:
            w = r.choice(NOVEL["things"])
            p = r.choice([f"i like {w}", f"i love {w}", f"i really like {w}"])
            valid = [f"{w} is nice.", f"oh, i do not like {w}.", f"why do you like {w}?"]
            name = "novel_like"
        out.append((p, r.choice(valid), valid, name))
    return out


def render(exs: list[tuple[str, str, list[str], str]]) -> str:
    return "".join(f"you: {p}\nbot: {a}\n" for p, a, _, _ in exs)


def encode(text: str) -> list[int]:
    return [INDEX[c] for c in text]


def decode(ids) -> str:
    return "".join(ALPHABET[i] for i in ids)


def valid_reply(prompt: str, reply: str) -> bool | None:
    """Is ``reply`` one of the replies the generator would accept for ``prompt``?
    None when the prompt is not one the generator makes."""
    sets = valid_sets()
    if prompt in sets:
        return reply in sets[prompt]
    for p, _, valid, _ in novel_exchanges(400):
        if p == prompt:
            return reply in valid
    return None


_SETS: dict[str, set[str]] | None = None


def valid_sets() -> dict[str, set[str]]:
    global _SETS
    if _SETS is None:
        r = random.Random(12345)
        s: dict[str, set[str]] = {}
        for _ in range(200_000):
            p, valid, _ = intents(r)
            s.setdefault(p, set()).update(valid)
        _SETS = s
    return _SETS


# Fixed splits. Seeds differ, so held-out exchanges are fresh draws of the same grammar;
# many short prompts recur (there are only a few hundred distinct prompts), which
# ``overlap`` reports.
TRAIN_SEED, VAL_SEED, TEST_SEED = 1, 2, 3


def splits(train: int = 6000, val: int = 300, test: int = 300):
    return exchanges(train, TRAIN_SEED), exchanges(val, VAL_SEED), exchanges(test, TEST_SEED)


if __name__ == "__main__":
    tr, va, te = splits()
    text = render(tr)
    print(f"alphabet {V}, train chars {len(text)}, exchanges {len(tr)}")
    print(f"distinct prompts in grammar: {len(valid_sets())}")
    seen = {(p, a) for p, a, _, _ in tr}
    print(f"test exchanges seen verbatim in train: {sum((p, a) in seen for p, a, _, _ in te) / len(te):.2f}")
    print(render(tr[:8]))
    print(render(novel_exchanges(4)))
