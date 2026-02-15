'''# delete one
python -c "from users import delete_user; print(delete_user('alex@example.com'))"

# list users
python -c "from users import list_users; print(list_users())"

# change password
python -c "from users import set_password; print(set_password('alex@example.com','NewPass!42'))"


#Create an admin user (example):
python -c "from users import create_user; create_user('admin@example.com','StrongPass!23', True)"

#Create an non-admin user (example):
python -c "from users import create_user; create_user('user@example.com','StrongPass!23', False)"
'''

# users.py
from datetime import datetime
from argon2 import PasswordHasher
import sqlite3, os

DB = os.getenv("AUTH_DB_PATH", "users.db")
ph = PasswordHasher()

def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )""")

def create_user(email: str, password: str, is_admin: bool = False):
    init_db()
    with sqlite3.connect(DB) as c:
        c.execute("INSERT INTO users(email,password_hash,is_admin,created_at) VALUES (?,?,?,?)",
                  (email.lower().strip(), ph.hash(password), 1 if is_admin else 0, datetime.utcnow().isoformat()))

def verify_user(email: str, password: str) -> bool:
    init_db()
    with sqlite3.connect(DB) as c:
        row = c.execute("SELECT password_hash FROM users WHERE email=?", (email.lower().strip(),)).fetchone()
    if not row:
        return False
    try:
        ph.verify(row[0], password)
        return True
    except Exception:
        return False


def user_exists(email: str) -> bool:
    with sqlite3.connect(DB) as c:
        row = c.execute("SELECT 1 FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    return bool(row)

def delete_user(email: str) -> bool:
    with sqlite3.connect(DB) as c:
        cur = c.execute("DELETE FROM users WHERE email=?", (email.strip().lower(),))
        c.commit()
        return cur.rowcount > 0

def list_users():
    with sqlite3.connect(DB) as c:
        return c.execute("SELECT email, is_admin, created_at FROM users ORDER BY email").fetchall()

def set_password(email: str, new_password: str) -> bool:
    with sqlite3.connect(DB) as c:
        cur = c.execute(
            "UPDATE users SET password_hash=? WHERE email=?",
            (ph.hash(new_password), email.strip().lower())
        )
        c.commit()
        return cur.rowcount > 0
