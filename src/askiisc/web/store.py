"""Accounts, sessions, chats, the allowlist and admin actions in SQLite.

The file lives in the private folder (mode 700). Chat titles and message
text are encrypted with a server key (Fernet), so a copy of the database
alone reveals no conversation. Session cookies are stored only as SHA-256
hashes. Admins have no way to read chats: nothing here lists another
user's messages.

Schema versions (`PRAGMA user_version`), applied in order at start-up and
only ever adding:
  1  accounts, sessions, chats, messages, password_resets, admin_actions
  2  allowed_users (who may sign in), signin_flows and accounts.ms_oid /
     ms_tid (from the Microsoft sign-in tried in October 2026; unused,
     kept rather than dropped)

`password_resets` holds the single-use links that set a password (an
admin's invitation or a password reset), stored as SHA-256 hashes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cryptography.fernet import Fernet, InvalidToken

from askiisc.common.errors import AskIIScError, ConfigError, NotFound

Status = Literal["pending", "active", "disabled"]
Access = Literal["active", "revoked"]
MAX_FAILED_LOGINS = 5
LOCKOUT_S = 15 * 60.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    user_id TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    synthetic INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    approved_by TEXT,
    last_login REAL,
    failed INTEGER NOT NULL DEFAULT 0,
    locked_until REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
    chat_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title BLOB NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS chats_user ON chats (user_id, updated);
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content BLOB NOT NULL,
    meta TEXT NOT NULL,
    created REAL NOT NULL,
    rating INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS messages_chat ON messages (chat_id, created);
CREATE TABLE IF NOT EXISTS password_resets (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created REAL NOT NULL,
    expires REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS admin_actions (
    time REAL NOT NULL,
    admin TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL
);
"""
SCHEMA_VERSION = 2
_V2_TABLES = (
    """CREATE TABLE IF NOT EXISTS allowed_users (
    email TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('active', 'revoked')),
    added_by TEXT NOT NULL,
    added_at REAL NOT NULL,
    revoked_by TEXT,
    revoked_at REAL
)""",
    """CREATE TABLE IF NOT EXISTS signin_flows (
    flow_hash TEXT PRIMARY KEY,
    flow BLOB NOT NULL,
    expires REAL NOT NULL
)""",
)
_ACCOUNT_COLUMNS = (
    "user_id, status, synthetic, created, approved_by, last_login, failed, "
    "locked_until, password_hash != ''"
)


class AccountExists(AskIIScError):
    """Registration for a user id that already has an account."""


@dataclass(frozen=True)
class Account:
    """One login account. Department and role come from the directory."""

    user_id: str
    status: Status
    synthetic: bool
    created: float
    approved_by: str | None
    last_login: float | None
    failed: int
    locked_until: float
    has_password: bool = True


@dataclass(frozen=True)
class Allowed:
    """One allowlist entry: who may sign in."""

    email: str
    status: Access
    added_by: str
    added_at: float
    revoked_by: str | None
    revoked_at: float | None


@dataclass(frozen=True)
class Chat:
    """One conversation (title decrypted)."""

    chat_id: str
    title: str
    created: float
    updated: float


@dataclass(frozen=True)
class StoredMessage:
    """One chat message (content decrypted)."""

    message_id: str
    role: str
    content: str
    meta: dict[str, Any]
    created: float
    rating: int


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def load_key(path: Path) -> bytes:
    """Reads the encryption key, creating it (mode 600) on first use."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
        path.write_bytes(Fernet.generate_key())
    return path.read_bytes().strip()


class AppStore:
    """Thread-safe store; every method is one short transaction.

    Args:
        path: The SQLite file.
        key: A Fernet key (`load_key`).
    """

    def __init__(self, path: Path, key: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(path), check_same_thread=False, timeout=60.0
        )
        try:
            self._fernet = Fernet(key)
        except (ValueError, TypeError) as err:
            raise ConfigError(
                "the chat encryption key is not a valid Fernet key; restore "
                "chat.key from its backup (a new key cannot read old chats)"
            ) from err
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Brings an older file up to SCHEMA_VERSION; never drops data."""
        with self._lock, self._db:
            version = int(self._db.execute("PRAGMA user_version").fetchone()[0])
            if version < 2:  # noqa: PLR2004
                columns = {
                    row[1]
                    for row in self._db.execute("PRAGMA table_info(accounts)")
                }
                for column in ("ms_oid", "ms_tid"):
                    if column not in columns:
                        self._db.execute(
                            f"ALTER TABLE accounts ADD COLUMN {column} TEXT"
                        )
                for table in _V2_TABLES:
                    self._db.execute(table)
                self._db.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS accounts_microsoft "
                    "ON accounts (ms_tid, ms_oid) WHERE ms_oid IS NOT NULL"
                )
            if version < SCHEMA_VERSION:  # a newer release's file keeps its
                self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def schema_version(self) -> int:
        """The file's schema version."""
        row = self._one("PRAGMA user_version", ())
        return int(row[0]) if row else 0

    def _seal(self, text: str) -> bytes:
        return self._fernet.encrypt(text.encode())

    def _open(self, blob: bytes) -> str:
        try:
            return self._fernet.decrypt(blob).decode()
        except InvalidToken:
            return "[unreadable: encrypted with another key]"

    def _one(self, sql: str, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        with self._lock:
            row = self._db.execute(sql, args).fetchone()
        return None if row is None else tuple(row)

    def _all(self, sql: str, args: tuple[Any, ...] = ()) -> list[Any]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _write(self, sql: str, args: tuple[Any, ...]) -> int:
        with self._lock, self._db:
            return self._db.execute(sql, args).rowcount

    # -- accounts --------------------------------------------------------

    @staticmethod
    def _account(row: tuple[Any, ...]) -> Account:
        user, status, synthetic, created, by, last, failed, locked, has = row
        return Account(
            user_id=user,
            status=status,
            synthetic=bool(synthetic),
            created=created,
            approved_by=by,
            last_login=last,
            failed=failed,
            locked_until=locked,
            has_password=bool(has),
        )

    def account(self, user_id: str) -> Account | None:
        """Returns an account, or None."""
        row = self._one(
            f"SELECT {_ACCOUNT_COLUMNS} FROM accounts WHERE user_id = ?",  # noqa: S608  # nosec B608
            (user_id,),
        )
        return None if row is None else self._account(row)

    def password_hash(self, user_id: str) -> str | None:
        """Returns the stored password hash, or None."""
        row = self._one(
            "SELECT password_hash FROM accounts WHERE user_id = ?", (user_id,)
        )
        return None if row is None else str(row[0])

    def create_account(
        self,
        user_id: str,
        password_hash: str,
        status: Status,
        now: float,
        synthetic: bool = False,
        approved_by: str | None = None,
    ) -> Account:
        """Adds an account.

        Raises:
            AccountExists: If the user already has one.
        """
        try:
            self._write(
                "INSERT INTO accounts (user_id, password_hash, status, "
                "synthetic, created, approved_by) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, password_hash, status, synthetic, now, approved_by),
            )
        except sqlite3.IntegrityError as err:
            raise AccountExists(user_id) from err
        account = self.account(user_id)
        if account is None:  # pragma: no cover - just inserted
            raise NotFound(user_id)
        return account

    def set_status(
        self, user_id: str, status: Status, by: str | None = None
    ) -> None:
        """Approves, disables or re-enables an account.

        Raises:
            NotFound: If there is no such account.
        """
        changed = self._write(
            "UPDATE accounts SET status = ?, "
            "approved_by = COALESCE(?, approved_by), failed = 0, "
            "locked_until = 0 WHERE user_id = ?",
            (status, by, user_id),
        )
        if not changed:
            raise NotFound(user_id)
        if status != "active":
            self.end_sessions(user_id)

    def activate_pending(self) -> int:
        """Activates accounts left pending by the old approval step.

        Being in the directory (added by an admin) is now the approval.

        Returns:
            How many accounts were activated.
        """
        return self._write(
            "UPDATE accounts SET status = 'active' WHERE status = 'pending'",
            (),
        )

    def set_password(self, user_id: str, password_hash: str) -> None:
        """Replaces the password and ends every session."""
        self._write(
            "UPDATE accounts SET password_hash = ? WHERE user_id = ?",
            (password_hash, user_id),
        )
        self.end_sessions(user_id)

    def replace_hash(self, user_id: str, password_hash: str) -> None:
        """Stores a stronger hash of the same password (sessions kept)."""
        self._write(
            "UPDATE accounts SET password_hash = ? WHERE user_id = ?",
            (password_hash, user_id),
        )

    def login_failed(self, user_id: str, now: float) -> None:
        """Counts a failed login; locks after MAX_FAILED_LOGINS in a row."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE accounts SET failed = failed + 1 WHERE user_id = ?",
                (user_id,),
            )
            self._db.execute(
                "UPDATE accounts SET locked_until = ?, failed = 0 "
                "WHERE user_id = ? AND failed >= ?",
                (now + LOCKOUT_S, user_id, MAX_FAILED_LOGINS),
            )

    def login_succeeded(self, user_id: str, now: float) -> None:
        """Clears the failure count and records the login time."""
        self._write(
            "UPDATE accounts SET failed = 0, last_login = ? WHERE user_id = ?",
            (now, user_id),
        )

    def accounts(self) -> list[Account]:
        """Every account, oldest first."""
        rows = self._all(
            f"SELECT {_ACCOUNT_COLUMNS} FROM accounts ORDER BY created"  # noqa: S608  # nosec B608
        )
        return [self._account(tuple(r)) for r in rows]

    # -- allowlist -------------------------------------------------------

    @staticmethod
    def _allowed(row: tuple[Any, ...]) -> Allowed:
        email, status, by, at, revoked_by, revoked_at = row
        return Allowed(email, status, by, at, revoked_by, revoked_at)

    def allowed(self, email: str) -> Allowed | None:
        """The allowlist entry of an e-mail address, or None."""
        row = self._one(
            "SELECT * FROM allowed_users WHERE email = ?", (email.lower(),)
        )
        return None if row is None else self._allowed(row)

    def allowed_users(self) -> dict[str, Allowed]:
        """Every allowlist entry, by e-mail."""
        rows = self._all("SELECT * FROM allowed_users")
        return {r[0]: self._allowed(tuple(r)) for r in rows}

    @contextlib.contextmanager
    def allowing(
        self, emails: Iterable[str], by: str, now: float
    ) -> Iterator[None]:
        """Allows e-mails in a transaction committed only if the block ends.

        Each gets an account without a password (set through a link) if it
        has none. The block makes the matching change elsewhere (the users
        tree); if it raises, nothing is allowed or created. A revoked
        address is allowed again.
        """
        with self._lock, self._db:
            for email in emails:
                self._db.execute(
                    "INSERT OR IGNORE INTO accounts (user_id, password_hash, "
                    "status, synthetic, created, approved_by) "
                    "VALUES (?, '', 'active', 0, ?, ?)",
                    (email.lower(), now, by),
                )
                self._db.execute(
                    "INSERT INTO allowed_users (email, status, added_by, "
                    "added_at) VALUES (?, 'active', ?, ?) "
                    "ON CONFLICT (email) DO UPDATE SET status = 'active', "
                    "added_by = excluded.added_by, "
                    "added_at = excluded.added_at, "
                    "revoked_by = NULL, revoked_at = NULL",
                    (email.lower(), by, now),
                )
            yield

    def set_access(
        self, email: str, status: Access, by: str, now: float
    ) -> bool:
        """Revokes or restores an address; ends its sessions when revoked.

        Chats and the account are kept.

        Returns:
            Whether the entry changed.
        """
        if status == "revoked":
            changed = self._write(
                "UPDATE allowed_users SET status = 'revoked', revoked_by = ?, "
                "revoked_at = ? WHERE email = ? AND status = 'active'",
                (by, now, email.lower()),
            )
            self.end_sessions(email.lower())
        else:
            changed = self._write(
                "UPDATE allowed_users SET status = 'active', "
                "revoked_by = NULL, revoked_at = NULL "
                "WHERE email = ? AND status = 'revoked'",
                (email.lower(),),
            )
        return changed > 0

    def allow_existing(self, emails: Iterable[str], by: str, now: float) -> int:
        """Allows addresses not yet listed (never re-allows a revoked one).

        Returns:
            How many were added.
        """
        added = 0
        with self._lock, self._db:
            for email in emails:
                added += self._db.execute(
                    "INSERT OR IGNORE INTO allowed_users (email, status, "
                    "added_by, added_at) VALUES (?, 'active', ?, ?)",
                    (email.lower(), by, now),
                ).rowcount
        return added

    def delete_account(self, user_id: str) -> None:
        """Deletes an account with its sessions, chats and messages."""
        self.delete_all_chats(user_id)
        self.end_sessions(user_id)
        self._write("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
        self._write("DELETE FROM accounts WHERE user_id = ?", (user_id,))

    # -- sessions --------------------------------------------------------

    def start_session(self, user_id: str, now: float, ttl_s: float) -> str:
        """Creates a session; returns the cookie value (stored hashed)."""
        token = secrets.token_urlsafe(32)
        self._write(
            "INSERT INTO sessions VALUES (?, ?, ?)",
            (_hash(token), user_id, now + ttl_s),
        )
        return token

    def session_user(self, token: str, now: float) -> str | None:
        """The user of a live session, or None."""
        row = self._one(
            "SELECT user_id, expires FROM sessions WHERE token_hash = ?",
            (_hash(token),),
        )
        if row is None:
            return None
        if row[1] <= now:
            self.end_session(token)
            return None
        return str(row[0])

    def end_session(self, token: str) -> None:
        """Logs one session out."""
        self._write(
            "DELETE FROM sessions WHERE token_hash = ?", (_hash(token),)
        )

    def end_sessions(self, user_id: str) -> None:
        """Logs a user out everywhere."""
        self._write("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def live_sessions(self, now: float) -> int:
        """The number of unexpired sessions."""
        return len(self.session_users(now))

    def session_users(self, now: float) -> list[str]:
        """The user of every unexpired session (one entry per session)."""
        self._write("DELETE FROM sessions WHERE expires <= ?", (now,))
        return [
            str(r[0]) for r in self._all("SELECT user_id FROM sessions", ())
        ]

    # -- password links --------------------------------------------------

    def create_link(self, user_id: str, now: float, ttl_s: float) -> str:
        """A single-use link token that sets the user's password.

        Earlier links of the user stop working; only the hash is stored.
        """
        token = secrets.token_urlsafe(32)
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM password_resets WHERE user_id = ?", (user_id,)
            )
            self._db.execute(
                "INSERT INTO password_resets VALUES (?, ?, ?, ?)",
                (_hash(token), user_id, now, now + ttl_s),
            )
        return token

    def actions_since(self, target: str, action: str, since: float) -> int:
        """How often an action was logged for a target since a time."""
        row = self._one(
            "SELECT count(*) FROM admin_actions "
            "WHERE target = ? AND action = ? AND time >= ?",
            (target, action, since),
        )
        return int(row[0]) if row else 0

    def link_user(self, token: str, now: float) -> str | None:
        """The user of an unexpired, unused link, or None."""
        row = self._one(
            "SELECT user_id, expires FROM password_resets WHERE token_hash = ?",
            (_hash(token),),
        )
        return None if row is None or row[1] <= now else str(row[0])

    def use_link(self, user_id: str, password_hash: str) -> None:
        """Sets the password from a link.

        Also ends every session, unlocks the account and voids all of the
        user's links, in one transaction.
        """
        with self._lock, self._db:
            self._db.execute(
                "UPDATE accounts SET password_hash = ?, failed = 0, "
                "locked_until = 0 WHERE user_id = ?",
                (password_hash, user_id),
            )
            self._db.execute(
                "DELETE FROM password_resets WHERE user_id = ?", (user_id,)
            )
            self._db.execute(
                "DELETE FROM sessions WHERE user_id = ?", (user_id,)
            )

    # -- chats -----------------------------------------------------------

    def create_chat(self, user_id: str, title: str, now: float) -> Chat:
        """Starts a chat."""
        chat_id = uuid.uuid4().hex
        self._write(
            "INSERT INTO chats VALUES (?, ?, ?, ?, ?)",
            (chat_id, user_id, self._seal(title), now, now),
        )
        return Chat(chat_id, title, now, now)

    def chats(self, user_id: str) -> list[Chat]:
        """A user's chats, most recently used first."""
        rows = self._all(
            "SELECT chat_id, title, created, updated FROM chats "
            "WHERE user_id = ? ORDER BY updated DESC",
            (user_id,),
        )
        return [Chat(c, self._open(t), a, u) for c, t, a, u in rows]

    def chat(self, user_id: str, chat_id: str) -> Chat:
        """One of the user's chats.

        Raises:
            NotFound: If it does not exist or belongs to someone else.
        """
        row = self._one(
            "SELECT chat_id, title, created, updated FROM chats "
            "WHERE chat_id = ? AND user_id = ?",
            (chat_id, user_id),
        )
        if row is None:
            raise NotFound(chat_id)
        return Chat(row[0], self._open(row[1]), row[2], row[3])

    def rename_chat(self, user_id: str, chat_id: str, title: str) -> None:
        """Renames a chat.

        Raises:
            NotFound: If it is not the user's chat.
        """
        if not self._write(
            "UPDATE chats SET title = ? WHERE chat_id = ? AND user_id = ?",
            (self._seal(title), chat_id, user_id),
        ):
            raise NotFound(chat_id)

    def touch_chat(self, chat_id: str, now: float) -> None:
        """Moves a chat to the top of the list."""
        self._write(
            "UPDATE chats SET updated = ? WHERE chat_id = ?", (now, chat_id)
        )

    def delete_chat(self, user_id: str, chat_id: str) -> None:
        """Deletes a chat and its messages.

        Raises:
            NotFound: If it is not the user's chat.
        """
        self.chat(user_id, chat_id)
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM messages WHERE chat_id = ?", (chat_id,)
            )
            self._db.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))

    def delete_all_chats(self, user_id: str) -> int:
        """Deletes every chat of a user; returns how many."""
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM messages WHERE chat_id IN "
                "(SELECT chat_id FROM chats WHERE user_id = ?)",
                (user_id,),
            )
            return self._db.execute(
                "DELETE FROM chats WHERE user_id = ?", (user_id,)
            ).rowcount

    # -- messages --------------------------------------------------------

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        meta: dict[str, Any],
        now: float,
    ) -> str:
        """Appends a message; returns its id."""
        message_id = uuid.uuid4().hex
        self._write(
            "INSERT INTO messages (message_id, chat_id, role, content, meta, "
            "created) VALUES (?, ?, ?, ?, ?, ?)",
            (
                message_id,
                chat_id,
                role,
                self._seal(content),
                json.dumps(meta),
                now,
            ),
        )
        self.touch_chat(chat_id, now)
        return message_id

    def messages(self, chat_id: str) -> list[StoredMessage]:
        """A chat's messages, oldest first."""
        rows = self._all(
            "SELECT message_id, role, content, meta, created, rating "
            "FROM messages WHERE chat_id = ? ORDER BY created, rowid",
            (chat_id,),
        )
        return [
            StoredMessage(m, r, self._open(c), json.loads(meta), t, rating)
            for m, r, c, meta, t, rating in rows
        ]

    def delete_message(self, chat_id: str, message_id: str) -> None:
        """Removes one message (used to regenerate an answer)."""
        self._write(
            "DELETE FROM messages WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id),
        )

    def rate(self, user_id: str, message_id: str, rating: int) -> None:
        """Thumbs up (1), down (-1) or clear (0) on the user's own answer.

        Raises:
            NotFound: If the message is not an answer in the user's chats.
        """
        if not self._write(
            "UPDATE messages SET rating = ? WHERE message_id = ? "
            "AND role = 'assistant' AND chat_id IN "
            "(SELECT chat_id FROM chats WHERE user_id = ?)",
            (rating, message_id, user_id),
        ):
            raise NotFound(message_id)

    def ratings(self, user_ids: set[str] | None = None) -> dict[str, int]:
        """Counts of thumbs up and down (no text), optionally per users."""
        rows = self._all(
            "SELECT c.user_id, m.rating FROM messages m "
            "JOIN chats c ON c.chat_id = m.chat_id WHERE m.rating != 0"
        )
        out = {"up": 0, "down": 0}
        for user, rating in rows:
            if user_ids is None or user in user_ids:
                out["up" if rating > 0 else "down"] += 1
        return out

    # -- admin actions ---------------------------------------------------

    def log_action(
        self, admin: str, action: str, target: str, now: float
    ) -> None:
        """Records something an admin did or looked at."""
        self._write(
            "INSERT INTO admin_actions VALUES (?, ?, ?, ?)",
            (now, admin, action, target),
        )

    def actions(
        self, limit: int = 200, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Admin actions, newest first, one page at a time."""
        rows = self._all(
            "SELECT time, admin, action, target FROM admin_actions "
            "ORDER BY time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [
            {"time": t, "admin": a, "action": act, "target": tg}
            for t, a, act, tg in rows
        ]

    def action_count(self) -> int:
        """How many admin actions are recorded."""
        row = self._one("SELECT count(*) FROM admin_actions", ())
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Closes the database."""
        self._db.close()
