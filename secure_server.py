# secure_server.py
from dotenv import load_dotenv
load_dotenv()

import os, re, urllib.parse
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from jose import jwt
import gradio as gr
from users import init_db, verify_user

# ====== CONFIG ======
SECRET = os.getenv("AUTH_SECRET", "change-me")         # set a strong secret in production
JWT_ALG = "HS256"
SESSION_MIN = int(os.getenv("SESSION_MIN", "360"))     # 3 hours
COOKIE_NAME = "session"
JWT_LEEWAY_SEC = int(os.getenv("JWT_LEEWAY_SEC", "900"))  # NEW: 15 min leeway

print("[debug] FastAPI AUTH_SECRET prefix:", SECRET[:8])

# 👇 ensure the Gradio module sees the exact same secret
os.environ["AUTH_SECRET"] = SECRET

ALLOWLIST_PREFIXES = (
    "/app/gradio_api/queue",
    "/app/file",
    "/app/assets",
    "/app/theme.css",
    "/app/favicon.ico",
    "/app/healthz",
)


# ====== AUTH HELPERS ======
def make_token(email: str) -> str:
    now = datetime.utcnow()
    return jwt.encode(
        {"sub": email, "iat": int(now.timestamp()), "exp": int((now + timedelta(minutes=SESSION_MIN)).timestamp())},
        SECRET, algorithm=JWT_ALG
    )

def current_user(req: Request):
    import time  # ensure available
    tok = req.cookies.get(COOKIE_NAME)
    if not tok:
        print(f"[auth] no '{COOKIE_NAME}' cookie on {req.url.path}")
        return None
    try:
        # Verify signature, but skip built-in exp check (we'll apply manual leeway)
        payload = jwt.decode(
            tok,
            SECRET,
            algorithms=[JWT_ALG],
            options={"verify_exp": False},  # <-- no 'leeway' kw (not supported in your jose)
        )

        # Manual leeway on exp
        exp = payload.get("exp")
        now = int(time.time())
        if isinstance(exp, (int, float)):
            if now > int(exp) + JWT_LEEWAY_SEC:
                print(f"[auth] token expired (now={now}, exp={exp}, leeway={JWT_LEEWAY_SEC}) on {req.url.path}")
                return None

        sub = payload.get("sub")
        print(f"[auth] token OK for {sub} on {req.url.path}")
        return sub

    except Exception as e:
        print(f"[auth] token decode failed on {req.url.path}: {type(e).__name__}: {e}")
        return None



# ====== FASTAPI APP ======
app = FastAPI()
init_db()

LOGIN_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Login</title>
<style>body{font-family:sans-serif;max-width:420px;margin:6rem auto}input{width:100%;padding:10px;margin:8px 0}button{padding:10px 14px}</style>
</head><body>
<h2>Hardware QA Assistant — Sign in</h2>
<form method="post" action="/login">
  <input type="email" name="email" placeholder="email" required />
  <input type="password" name="password" placeholder="password" required />
  <button type="submit">Sign in</button>
</form>
</body></html>
"""

@app.get("/login", response_class=HTMLResponse)
async def login_page(req: Request):
    if current_user(req):
        return RedirectResponse(url="/")
    return HTMLResponse(LOGIN_HTML)

@app.post("/login")
async def login(req: Request):
    form = await req.form()
    email = str(form.get("email", "")).strip().lower()
    password = str(form.get("password", ""))
    if not verify_user(email, password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    tok = make_token(email)
    resp = RedirectResponse(url=f"/app/?ts={int(datetime.utcnow().timestamp())}", status_code=302)
    # HttpOnly JWT cookie for auth (Path=/ so it's sent to ALL /app/* requests)
    resp.set_cookie(COOKIE_NAME, tok, httponly=True, samesite="lax", secure=False,
                    max_age=SESSION_MIN * 60, path="/")
    # Non-HttpOnly display cookie so UI can show the user label even if it can't read the JWT
    resp.set_cookie("who", email, httponly=False, samesite="lax", secure=False,
                    max_age=SESSION_MIN * 60, path="/")
    return resp



@app.get("/logout")
@app.post("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie(COOKIE_NAME, path="/")
    resp.delete_cookie("who", path="/")
    return resp

@app.get("/whoami")
async def whoami(req: Request):
    return {"user": current_user(req)}



from users import init_db, verify_user, user_exists  # ensure user_exists is imported

@app.middleware("http")
async def auth_guard(request: Request, call_next):
    path = request.url.path

    # Allow Gradio queue bootstrap calls (harmless, no private data)
    if path.startswith("/app/gradio_api/queue"):
        return await call_next(request)

    # Protect everything else under /app
    if path.startswith("/app"):
        try:
            from users import user_exists
        except Exception:
            user_exists = lambda _: True

        u = current_user(request)
        if not u or not user_exists(u):
            print(f"[guard] redirect → /login from {path} (user={u})")
            return RedirectResponse(url="/login", status_code=302)

    # Normal flow
    resp = await call_next(request)

    # Avoid caching the shell
    if path in ("/app", "/app/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp




@app.get("/")
async def root(req: Request):
    # send anonymous users to /login, authenticated users to /app
    return RedirectResponse(url="/app" if current_user(req) else "/login", status_code=302)


# ====== GRADIO MOUNT ======
import hardware_qa_assistant_lang_graph_gradio_demo_version_users as demo_mod
demo = demo_mod.build_gradio_blocks(auth_secret=SECRET)  # pass the same key used to sign JWT
app = gr.mount_gradio_app(app, demo, path="/app")


