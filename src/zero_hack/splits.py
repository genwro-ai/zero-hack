import hashlib

SPLIT_NAMES = ("train", "valid", "test")


def split_for(family: str, steps: list[str]) -> str:
    """Assign one sequence to a split using a stable hash of (family, steps).

    Buckets a sha1 digest into 0-99: 0-79 -> train, 80-89 -> valid, 90-99 -> test
    (an 80 / 10 / 10 split).
    """
    payload = family.lower() + "\x00" + "\n".join(steps)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"
