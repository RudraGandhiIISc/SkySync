"""Account administration from the command line (run inside Slurm).

    python -m askiisc.web.cli list
    python -m askiisc.web.cli add-institute-admin USER_ID
    python -m askiisc.web.cli remove-institute-admin USER_ID
    python -m askiisc.web.cli disable USER_ID     # also: enable
    python -m askiisc.web.cli allow USER_ID       # also: revoke
    python -m askiisc.web.cli password-link USER_ID
    python -m askiisc.web.cli seed-locust         # load-test accounts

Institute admins are added only here, never from the dashboard (there is
no web setup page). Adding one puts them in the directory and on the
allowlist with an account without a password; they then use Create
account on the sign-in page. Adding someone who is already there changes
nothing. `password-link` prints a single-use link (1 hour) that sets the
password of anyone in the directory with an account: how a locked-out
institute admin gets back in, or the first one sets a password while
e-mail is not set up.
`allow` and `revoke` change only the allowlist of someone already in the
directory (revoking keeps their account and chats). Links are printed
once, in the terminal of the person who ran the command; run it with
`srun --pty` so they do not stay in a job log.

`seed-locust` gives every test user in the directory (ids without "@") a
synthetic account with one shared random password, written to
locust_accounts.json (mode 600) for Locust. Real users (e-mail ids) never
get synthetic accounts.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from pathlib import Path

from askiisc.access.directory import Directory, Person
from askiisc.access.policy import INSTITUTE
from askiisc.common.config import Settings
from askiisc.common.errors import NotFound
from askiisc.common.types import UserId
from askiisc.web.dbfiles import db_dir, restore
from askiisc.web.passwords import hash_password
from askiisc.web.people import PeopleTree, make_entry
from askiisc.web.store import AccountExists, AppStore, load_key

SAMPLE_USERS = Path("eval/datasets/sample/users")


def is_test_user(person: Person) -> bool:
    """Test users have plain ids; real users are e-mail addresses."""
    return "@" not in person.user_id


def seed_locust(
    store: AppStore, people: list[Person], out: Path, now: float
) -> int:
    """Creates synthetic accounts and the Locust credentials file.

    Returns:
        The number of synthetic accounts in the file.
    """
    password = secrets.token_urlsafe(18)
    hashed = hash_password(password)
    users = []
    for person in sorted(people, key=lambda p: p.user_id):
        if not is_test_user(person):
            continue
        existing = store.account(person.user_id)
        if existing is not None and not existing.synthetic:
            continue  # someone registered this test user by hand
        try:
            store.create_account(
                person.user_id,
                hashed,
                "active",
                now,
                synthetic=True,
                approved_by="seed-locust",
            )
        except AccountExists:
            store.set_password(person.user_id, hashed)
            store.set_status(person.user_id, "active", "seed-locust")
        users.append(
            {
                "user_id": person.user_id,
                "department": person.department,
                "role": person.role,
                "groups": sorted(person.groups),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.touch(mode=0o600)
    out.write_text(json.dumps({"password": password, "users": users}, indent=1))
    return len(users)


def institute_admin(
    tree: PeopleTree, store: AppStore, user_id: str, add: bool, now: float
) -> str:
    """Adds or removes an institute admin in the tree and the allowlist.

    Returns:
        The normalised user id.

    Raises:
        ConfigError: On a bad id, or if the user is already in the tree
            (the allowlist is then unchanged).
        NotFound: On removing someone who is not an institute admin.
    """
    entry = make_entry(user_id, INSTITUTE, "admin")
    if add:
        with store.allowing([entry.user_id], "cli", now):
            tree.add([entry])
        return entry.user_id
    found = tree.find(entry.user_id)
    if found is None or (found.department, found.role) != (INSTITUTE, "admin"):
        raise NotFound(f"{entry.user_id} is not an institute admin")
    tree.remove(entry.user_id)
    store.set_access(entry.user_id, "revoked", "cli", now)
    return entry.user_id


def link_line(store: AppStore, app: Path, user_id: str) -> str:
    """Makes a single-use password link and the line that shows it."""
    from askiisc.web.password_links import (  # noqa: PLC0415
        TERMINAL_LINK_TTL_S,
        link_path,
    )

    base = "http://localhost:18000"
    smtp = app / "smtp.json"
    if smtp.exists():
        base = str(json.loads(smtp.read_text()).get("public_url") or base)
    ttl = TERMINAL_LINK_TTL_S
    token = store.create_link(user_id, time.time(), ttl)
    hours = int(ttl // 3600)
    return (
        f"single-use link for {user_id} (valid {hours} h; open it through "
        f"the SSH tunnel):\n{base.rstrip('/')}{link_path(token)}"
    )


def main() -> None:  # noqa: PLR0912, PLR0915  # pragma: no cover - one command table
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "list",
            "add-institute-admin",
            "remove-institute-admin",
            "disable",
            "enable",
            "allow",
            "revoke",
            "password-link",
            "seed-locust",
        ],
    )
    parser.add_argument("user_id", nargs="?")
    args = parser.parse_args()
    data = Path(os.environ.get("ASKIISC_DATA", "/data4/muthangiv/askiisc"))
    app = data / "private" / "app"
    restore(app, db_dir(app))  # never start from an empty local copy
    store = AppStore(db_dir(app) / "app.db", load_key(app / "chat.key"))
    settings = Settings()
    directory = Directory()
    users = (
        Path(settings.directory_path)
        if settings.directory_path
        else SAMPLE_USERS
    )
    directory.load(users)
    if args.command.endswith("institute-admin"):
        if not args.user_id:
            parser.error("give a user id (IISc e-mail)")
        add = args.command.startswith("add")
        uid = institute_admin(
            PeopleTree(users), store, args.user_id, add, time.time()
        )
        action = "institute_admin_" + ("add" if add else "remove")
        store.log_action("cli", action, uid, time.time())
        if not add:
            store.end_sessions(uid)
        print(f"{uid}: {'added' if add else 'removed'}")
        if add:
            print(
                "they create their account with Create account on the "
                "sign-in page"
            )
    elif args.command == "password-link":
        uid = (args.user_id or "").strip().lower()
        account = store.account(uid)
        if directory.get(UserId(uid)) is None or account is None:
            parser.error(
                "give the id of someone in the directory with an account"
            )
        store.log_action("cli", "password_link", uid, time.time())
        print(link_line(store, app, uid))
    elif args.command == "list":
        for a in store.accounts():
            kind = "locust" if a.synthetic else "real"
            print(f"{a.user_id}\t{a.status}\t{kind}")
    elif args.command == "seed-locust":
        count = seed_locust(
            store, directory.people(), app / "locust_accounts.json", time.time()
        )
        print(f"{count} synthetic accounts; credentials in {app}")
    elif args.command in {"allow", "revoke"}:
        uid = (args.user_id or "").strip().lower()
        if "@" not in uid or directory.get(UserId(uid)) is None:
            parser.error("give the e-mail address of someone in the directory")
        if args.command == "allow" and store.allowed(uid) is None:
            store.allow_existing([uid], "cli", time.time())
        else:
            access = "active" if args.command == "allow" else "revoked"
            store.set_access(uid, access, "cli", time.time())  # type: ignore[arg-type]
        store.log_action("cli", f"access_{args.command}", uid, time.time())
        print(f"{uid}: {'allowed' if args.command == 'allow' else 'revoked'}")
    else:
        if not args.user_id or directory.get(UserId(args.user_id)) is None:
            parser.error("give a user id that is in the directory")
        status = "active" if args.command == "enable" else "disabled"
        store.set_status(args.user_id, status, "cli")  # type: ignore[arg-type]
        store.log_action(
            "cli", f"user_{args.command}", args.user_id, time.time()
        )
        print(f"{args.user_id}: {status}")


if __name__ == "__main__":  # pragma: no cover
    main()
