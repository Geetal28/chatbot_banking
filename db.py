"""Synthetic retail-banking database plus safe query helpers.

All data here is generated. No real customer data should ever be loaded into this demo.

Account questions are answered by LLM-generated SQL, which is risky by nature, so the
safety design is layered:
  1. The query runs against a throw-away in-memory copy that contains ONLY the current
     customer's rows, so cross-customer access is impossible by construction.
  2. A SQLite authorizer allows only SELECT, reads of whitelisted tables and a short
     list of functions. Anything else is denied at compile time.
  3. One statement only, a row cap and an operation-count cap against runaway queries.
"""
from __future__ import annotations

import random
import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    segment     TEXT NOT NULL
);
CREATE TABLE accounts (
    account_id  TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    type        TEXT NOT NULL,          -- current | savings | credit_card
    balance     REAL NOT NULL,          -- GBP, credit card balance is negative when owed
    opened      TEXT NOT NULL
);
CREATE TABLE transactions (
    txn_id      INTEGER PRIMARY KEY,
    account_id  TEXT NOT NULL REFERENCES accounts(account_id),
    date        TEXT NOT NULL,          -- YYYY-MM-DD
    description TEXT NOT NULL,
    category    TEXT NOT NULL,
    amount      REAL NOT NULL           -- GBP, negative = money out, positive = money in
);
"""

# Shown to the LLM for text-to-SQL. Keep in sync with SCHEMA.
SCHEMA_FOR_LLM = """Tables (SQLite):
accounts(account_id TEXT, customer_id TEXT, type TEXT /* current|savings|credit_card */, balance REAL /* GBP */, opened TEXT /* YYYY-MM-DD */)
transactions(txn_id INTEGER, account_id TEXT, date TEXT /* YYYY-MM-DD */, description TEXT, category TEXT, amount REAL /* GBP, negative = spending/money out, positive = money in */)
Categories: groceries, dining, transport, utilities, rent, entertainment, shopping, salary, transfer, interest, subscriptions, health
"""

ALLOWED_TABLES = {"accounts", "transactions"}
ALLOWED_FUNCTIONS = {
    "sum", "count", "avg", "min", "max", "total", "round", "abs", "date", "strftime",
    "lower", "upper", "coalesce", "ifnull", "substr", "length", "cast", "printf",
    "julianday", "iif", "like",
}

_CUSTOMERS = [
    ("C1001", "Alex Demo", "everyday"),
    ("C1002", "Sam Sample", "saver"),
    ("C1003", "Jo Placeholder", "everyday"),
]

# Fictional merchants on purpose: no real brands in synthetic data.
_MERCHANTS = {
    "groceries": ["Greenfield Grocers", "Corner Market", "Fresh Basket"],
    "dining": ["Riverside Cafe", "Noodle House", "Pizza Corner"],
    "transport": ["City Transit", "Rail Pass Co", "Fuel Stop"],
    "utilities": ["Power & Light Co", "Water Board", "Broadband Plus"],
    "entertainment": ["Cinema Central", "Game Store", "Concert Tickets"],
    "shopping": ["Home Goods Ltd", "Fashion Row", "Tech Outlet"],
    "subscriptions": ["Streaming Service A", "Music Service B", "Gym Membership"],
    "health": ["High Street Pharmacy", "Dental Practice"],
}


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path, seed: int = 42, force: bool = False) -> None:
    """Create and populate the demo DB. No-op if it already exists unless force=True."""
    path = Path(path)
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    rng = random.Random(seed)
    today = date.today()
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO customers VALUES (?,?,?)", _CUSTOMERS)

    txn_id = 1
    for idx, (cid, _name, segment) in enumerate(_CUSTOMERS):
        accounts = [(f"{cid}-CUR", "current", 1200.0 + 400 * idx), (f"{cid}-SAV", "savings", 3000.0 + 1500 * idx)]
        if segment == "everyday":
            accounts.append((f"{cid}-CC", "credit_card", 0.0))
        for acc_id, acc_type, opening in accounts:
            rows: list[tuple[Any, ...]] = []
            if acc_type == "current":
                for m in range(6):
                    month_start = (today.replace(day=1) - timedelta(days=30 * m)).replace(day=1)
                    rows.append((acc_id, (month_start + timedelta(days=0)).isoformat(), "Monthly salary", "salary", round(2400 + 150 * idx, 2)))
                    rows.append((acc_id, (month_start + timedelta(days=1)).isoformat(), "Rent payment", "rent", -round(850 + 50 * idx, 2)))
                for _ in range(70):
                    cat = rng.choice(["groceries", "groceries", "dining", "transport", "utilities", "entertainment", "shopping", "subscriptions", "health"])
                    d = today - timedelta(days=rng.randint(0, 180))
                    amt = -round(rng.uniform(3, 95), 2) if cat != "utilities" else -round(rng.uniform(25, 110), 2)
                    rows.append((acc_id, d.isoformat(), rng.choice(_MERCHANTS[cat]), cat, amt))
            elif acc_type == "savings":
                for m in range(6):
                    month_start = (today.replace(day=1) - timedelta(days=30 * m)).replace(day=1)
                    rows.append((acc_id, (month_start + timedelta(days=2)).isoformat(), "Transfer from current", "transfer", 200.0))
                    rows.append((acc_id, (month_start + timedelta(days=27 if month_start.month != 2 else 26)).isoformat(), "Interest paid", "interest", round(rng.uniform(4, 15), 2)))
            else:  # credit card
                for _ in range(30):
                    cat = rng.choice(["shopping", "dining", "entertainment", "subscriptions"])
                    d = today - timedelta(days=rng.randint(0, 90))
                    rows.append((acc_id, d.isoformat(), rng.choice(_MERCHANTS[cat]), cat, -round(rng.uniform(8, 120), 2)))
            rows = [r for r in rows if r[1] <= today.isoformat()]
            balance = round(opening + sum(r[4] for r in rows), 2)
            conn.execute(
                "INSERT INTO accounts VALUES (?,?,?,?,?)",
                (acc_id, cid, acc_type, balance, (today - timedelta(days=900 + 100 * idx)).isoformat()),
            )
            for r in sorted(rows, key=lambda x: x[1]):
                conn.execute("INSERT INTO transactions VALUES (?,?,?,?,?,?)", (txn_id, *r))
                txn_id += 1
    conn.commit()
    conn.close()


def list_customers(path: Path) -> list[dict[str, str]]:
    conn = connect(path)
    try:
        return [dict(r) for r in conn.execute("SELECT customer_id, name, segment FROM customers ORDER BY customer_id")]
    finally:
        conn.close()


def get_as_of(path: Path, customer_id: str) -> str:
    """Latest transaction date for this customer, used as 'today' in prompts."""
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT MAX(t.date) FROM transactions t JOIN accounts a ON a.account_id=t.account_id WHERE a.customer_id=?",
            (customer_id,),
        ).fetchone()
        return row[0] or date.today().isoformat()
    finally:
        conn.close()


# ---------- Deterministic tools (parameterised SQL, no LLM involved) ----------

def get_accounts(path: Path, customer_id: str) -> list[dict[str, Any]]:
    conn = connect(path)
    try:
        return [dict(r) for r in conn.execute(
            "SELECT account_id, type, balance FROM accounts WHERE customer_id=? ORDER BY account_id", (customer_id,))]
    finally:
        conn.close()


def get_recent_transactions(path: Path, customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
    conn = connect(path)
    try:
        return [dict(r) for r in conn.execute(
            "SELECT t.date, a.type AS account, t.description, t.category, t.amount "
            "FROM transactions t JOIN accounts a ON a.account_id=t.account_id "
            "WHERE a.customer_id=? ORDER BY t.date DESC, t.txn_id DESC LIMIT ?",
            (customer_id, int(limit)))]
    finally:
        conn.close()


def spending_by_category(path: Path, customer_id: str, days: int = 30) -> list[dict[str, Any]]:
    as_of = get_as_of(path, customer_id)
    conn = connect(path)
    try:
        return [dict(r) for r in conn.execute(
            "SELECT t.category, ROUND(-SUM(t.amount), 2) AS spent "
            "FROM transactions t JOIN accounts a ON a.account_id=t.account_id "
            "WHERE a.customer_id=? AND t.amount<0 AND t.date > date(?, ?) "
            "AND t.category NOT IN ('rent','transfer') "
            "GROUP BY t.category ORDER BY spent DESC",
            (customer_id, as_of, f"-{int(days)} days"))]
    finally:
        conn.close()


# ---------- Safe LLM-generated SQL ----------

class UnsafeSQL(ValueError):
    pass


def _scoped_memory_db(path: Path, customer_id: str) -> sqlite3.Connection:
    """In-memory DB holding only this customer's accounts and transactions."""
    src = connect(path)
    mem = sqlite3.connect(":memory:")
    mem.row_factory = sqlite3.Row
    try:
        mem.executescript(SCHEMA)
        accs = src.execute("SELECT * FROM accounts WHERE customer_id=?", (customer_id,)).fetchall()
        ids = [a["account_id"] for a in accs]
        mem.executemany("INSERT INTO accounts VALUES (?,?,?,?,?)", [tuple(a) for a in accs])
        if ids:
            q = ",".join("?" * len(ids))
            txs = src.execute(f"SELECT * FROM transactions WHERE account_id IN ({q})", ids).fetchall()
            mem.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?)", [tuple(t) for t in txs])
        mem.commit()
    finally:
        src.close()
    return mem


def _authorizer(action: int, arg1: str | None, arg2: str | None, _db: str | None, _src: str | None) -> int:
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if (arg1 or "").lower() in ALLOWED_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if (arg2 or "").lower() in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def clean_sql(sql: str) -> str:
    sql = sql.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
    if fence:
        sql = fence.group(1).strip()
    sql = sql.rstrip(";").strip()
    if ";" in sql:
        raise UnsafeSQL("Only a single statement is allowed.")
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise UnsafeSQL("Only SELECT queries are allowed.")
    return sql


def run_readonly_sql(path: Path, customer_id: str, sql: str, max_rows: int = 50) -> tuple[list[str], list[tuple[Any, ...]]]:
    sql = clean_sql(sql)
    mem = _scoped_memory_db(path, customer_id)
    try:
        mem.set_authorizer(_authorizer)
        budget = {"ops": 0}

        def progress() -> int:
            budget["ops"] += 1
            return 1 if budget["ops"] > 2000 else 0  # abort long-running queries

        mem.set_progress_handler(progress, 1000)
        try:
            cur = mem.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows = [tuple(r) for r in cur.fetchmany(max_rows)]
        except sqlite3.DatabaseError as e:
            raise UnsafeSQL(f"Query rejected or failed: {e}") from e
        return cols, rows
    finally:
        mem.close()
