"""Password hashing: Argon2id (argon2-cffi), with old scrypt hashes upgraded.

New hashes are Argon2id in the standard encoded form ("$argon2id$..."),
with argon2-cffi's default parameters (RFC 9106 low-memory profile). Hashes
made before Argon2 ("scrypt$<n>$<r>$<p>$<salt hex>$<hash hex>") still
verify; `needs_rehash` tells the caller to replace them at the next
successful sign-in. Passwords are never truncated: anything longer than
MAX_PASSWORD_CHARS is refused.
"""

from __future__ import annotations

import functools
import hashlib
import hmac

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from askiisc.common.errors import AuthError

MIN_PASSWORD_CHARS = 8
MAX_PASSWORD_CHARS = 256
_HASHER = PasswordHasher()


def check_strength(password: str, user_id: str) -> None:
    """Rejects passwords that are too short, too long or the user id.

    Raises:
        AuthError: With the reason, for the person choosing it.
    """
    if not MIN_PASSWORD_CHARS <= len(password) <= MAX_PASSWORD_CHARS:
        raise AuthError(
            f"The password needs at least {MIN_PASSWORD_CHARS} characters "
            f"(at most {MAX_PASSWORD_CHARS})."
        )
    if password.strip().lower() in {
        user_id.lower(),
        user_id.split("@", 1)[0].lower(),
    }:
        raise AuthError("The password must not be your e-mail address.")


def hash_password(password: str) -> str:
    """An Argon2id hash with a fresh random salt."""
    return _HASHER.hash(password)


def verify_password(password: str, stored: str) -> bool:
    """Checks a password against a stored Argon2id or old scrypt hash."""
    if stored.startswith("scrypt$"):
        return _verify_scrypt(password, stored)
    try:
        return _HASHER.verify(stored, password)
    except (VerificationError, InvalidHashError):
        return False


def needs_rehash(stored: str) -> bool:
    """Whether a verified hash should be replaced (old scheme or params)."""
    if not stored.startswith("$argon2"):
        return True
    return _HASHER.check_needs_rehash(stored)


@functools.cache
def _dummy() -> str:
    return _HASHER.hash("no account has this password")


def burn(password: str) -> None:
    """Spends the time of one verification, for ids with no account.

    Answers then take as long whether or not the account exists.
    """
    verify_password(password, _dummy())


def _verify_scrypt(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        got = hashlib.scrypt(
            password.encode(),
            salt=bytes.fromhex(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            maxmem=64 * 1024 * 1024,
        )
    except ValueError:
        return False
    return hmac.compare_digest(got.hex(), digest)
