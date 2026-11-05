"""Login check and tenant lookup for the gateway.

The user id comes from the verified SSO token. The tenant (department)
comes from the directory export, never from the client (A4).
"""

from __future__ import annotations

from dataclasses import dataclass

from askiisc.access.directory import Directory, Person
from askiisc.common.errors import AccessDenied, AuthError
from askiisc.common.tokens import TokenVerifier
from askiisc.common.types import TenantId, UserId


@dataclass(frozen=True)
class Caller:
    """An authenticated request's identity."""

    user_id: UserId
    tenant: TenantId
    token: str


def bearer_token(header: str | None) -> str:
    """Extracts the token from an Authorization header.

    Raises:
        AuthError: If the header is missing or not a Bearer token.
    """
    scheme, _, token = (header or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("missing bearer token")
    return token.strip()


class Authenticator:
    """Verifies tokens and resolves tenants."""

    def __init__(self, verifier: TokenVerifier, directory: Directory) -> None:
        self._verifier = verifier
        self._directory = directory

    def authenticate(self, header: str | None) -> Caller:
        """Authenticates a request.

        Args:
            header: The Authorization header value.

        Returns:
            The caller.

        Raises:
            AuthError: If the token is missing or invalid (HTTP 401).
            AccessDenied: If the user is not in the directory (HTTP 403).
        """
        token = bearer_token(header)
        user = self._verifier.verify(token)
        person = self._directory.get(user)
        if person is None:
            raise AccessDenied("user not in directory")
        return Caller(user_id=user, tenant=tenant_of(person), token=token)


def tenant_of(person: Person) -> TenantId:
    """A person's tenant: their department."""
    return TenantId(person.department or "unassigned")
