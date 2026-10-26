import hashlib


def stable_id(product_id: str) -> int:
    """
    63-bit FAISS id that is the same in every process.

    The previous code used Python's built-in hash(), which is randomised per
    process (PYTHONHASHSEED), so re-adding a product after a restart created a
    second vector instead of replacing the first.
    """
    digest = hashlib.blake2b(str(product_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF
