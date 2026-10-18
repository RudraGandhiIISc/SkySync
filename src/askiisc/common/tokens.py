"""Verification of institute SSO tokens (JWT, RS256).

Only the user id (`sub`) is taken from a token. Departments and roles come
from the directory export, never from the token or the client (A4).
Expiry is checked against the injected Clock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from askiisc.common.clock import Clock
from askiisc.common.errors import AuthError
from askiisc.common.types import UserId

ALGORITHM = "RS256"


class TokenVerifier:
    """Verifies RS256 tokens against a JWKS key set."""

    def __init__(
        self,
        jwks: dict[str, Any],
        issuer: str,
        audience: str,
        clock: Clock,
    ) -> None:
        self._keys = {
            key.key_id: key for key in jwt.PyJWKSet.from_dict(jwks).keys
        }
        self._issuer = issuer
        self._audience = audience
        self._clock = clock

    @classmethod
    def from_file(
        cls, path: Path, issuer: str, audience: str, clock: Clock
    ) -> TokenVerifier:
        """Loads the key set from a JWKS JSON file."""
        return cls(json.loads(path.read_text()), issuer, audience, clock)

    def verify(self, token: str) -> UserId:
        """Verifies a token and returns its user id.

        Args:
            token: The compact JWT.

        Returns:
            The `sub` claim.

        Raises:
            AuthError: If the token is malformed, wrongly signed, expired,
                for another issuer or audience, or has no subject.
        """
        try:
            kid = jwt.get_unverified_header(token).get("kid")
            key = self._keys[kid]
            claims = jwt.decode(
                token,
                key.key,
                algorithms=[ALGORITHM],
                audience=self._audience,
                issuer=self._issuer,
                options={"verify_exp": False, "require": ["exp", "sub"]},
            )
        except (jwt.PyJWTError, KeyError) as err:
            raise AuthError(f"invalid token: {err}") from err
        if float(claims["exp"]) <= self._clock.now():
            raise AuthError("token expired")
        subject = claims["sub"]
        if not isinstance(subject, str) or not subject:
            raise AuthError("token has no subject")
        return UserId(subject)


class TokenIssuer:
    """Signs tokens like the institute SSO. For tests and load tools only."""

    def __init__(
        self,
        kid: str = "test-key",
        issuer: str = "iisc-sso",
        audience: str = "ask-iisc",
    ) -> None:
        self._key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048
        )
        self.kid = kid
        self.issuer = issuer
        self.audience = audience

    def jwks(self) -> dict[str, Any]:
        """Returns the public key set."""
        public = json.loads(
            jwt.algorithms.RSAAlgorithm.to_jwk(self._key.public_key())
        )
        public.update({"kid": self.kid, "alg": ALGORITHM, "use": "sig"})
        return {"keys": [public]}

    def issue(self, user_id: str, expires_at: float, **extra: object) -> str:
        """Signs a token for a user.

        Args:
            user_id: The `sub` claim.
            expires_at: The `exp` claim, in seconds.
            **extra: Extra claims (ignored by the verifier).

        Returns:
            The compact JWT.
        """
        claims = {
            "sub": user_id,
            "exp": int(expires_at),
            "iss": self.issuer,
            "aud": self.audience,
            **extra,
        }
        return jwt.encode(
            claims, self._key, algorithm=ALGORITHM, headers={"kid": self.kid}
        )

    def verifier(self, clock: Clock) -> TokenVerifier:
        """Returns a verifier that trusts this issuer."""
        return TokenVerifier(self.jwks(), self.issuer, self.audience, clock)
