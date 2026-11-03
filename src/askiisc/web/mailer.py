"""Outgoing e-mail for password links (smtplib only).

Configured by `private/app/smtp.json` (mode 600), for example a Gmail
account with an app password:

    {"enabled": true, "host": "smtp.gmail.com", "port": 587,
     "starttls": true, "user": "<address>", "password": "<app password>",
     "from": "Ask IISc <address>", "public_url": "http://localhost:18000"}

A relay that takes mail without signing in sets "auth": false and leaves
out user and password.
`public_url` is the address people open the chat app at; links point
there. The password is never logged or shown.
"""

from __future__ import annotations

import json
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol

from askiisc.common.errors import ConfigError

SMTP_TIMEOUT_S = 20.0


class Mailer(Protocol):
    """Sends one plain-text e-mail."""

    public_url: str

    def send(self, to: str, subject: str, body: str) -> None:
        """Sends a message (blocking; call it off the event loop)."""


@dataclass(frozen=True)
class SmtpMailer:
    """An SMTP account or relay, with STARTTLS and a login by default."""

    host: str
    port: int
    sender: str
    public_url: str
    starttls: bool = True
    user: str = ""
    password: str = field(default="", repr=False)

    def send(self, to: str, subject: str, body: str) -> None:
        """Sends a message.

        Raises:
            OSError: If the server is unreachable or refuses the message
                (smtplib.SMTPException is an OSError).
        """
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)
        with smtplib.SMTP(self.host, self.port, timeout=SMTP_TIMEOUT_S) as smtp:
            if self.starttls:
                smtp.starttls()
            if self.user:
                smtp.login(self.user, self.password)
            smtp.send_message(message)


def load_smtp(path: Path) -> SmtpMailer | None:
    """The configured mailer, or None if e-mail is not set up.

    Raises:
        ConfigError: If the file is enabled but malformed or incomplete
            (names the missing fields, never their values).
    """
    if not path.exists():
        return None
    try:
        config = json.loads(path.read_text())
    except ValueError as err:
        raise ConfigError(f"{path.name} is not valid JSON") from err
    if not config.get("enabled"):
        return None
    needed = ["host", "from", "public_url"]
    if config.get("auth", True):
        needed += ["user", "password"]
    missing = [k for k in needed if not str(config.get(k, "")).strip()]
    if missing:
        raise ConfigError(f"{path.name} is enabled but lacks {missing}")
    return SmtpMailer(
        host=str(config["host"]),
        port=int(config.get("port", 587)),
        sender=str(config["from"]),
        public_url=str(config["public_url"]).rstrip("/"),
        starttls=bool(config.get("starttls", True)),
        user=str(config.get("user", "")),
        password=str(config.get("password", "")),
    )
