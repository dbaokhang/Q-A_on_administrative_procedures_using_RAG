"""
Vai trò và phân quyền (Bảng 2.1, BR-04, NFR-06).

Bản thử nghiệm mô phỏng tài khoản cục bộ (mục 2.1.1.1): data/registry/users.json lưu mật khẩu đã băm
PBKDF2-SHA256. Token phiên ký HMAC-SHA256 bằng APP_SECRET, có hạn dùng. Phạm vi dữ liệu (scopes) luôn do
backend suy ra từ vai trò đã xác thực — client không tự gửi phạm vi.

    python -m src.auth add-user canbo01 officer      # tạo / đổi mật khẩu tài khoản (hỏi mật khẩu)
    python -m src.auth list
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USERS_PATH = Path(os.getenv("USERS_FILE", PROJECT_ROOT / "data" / "registry" / "users.json"))
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", str(8 * 3600)))
PBKDF2_ROUNDS = 200_000

ROLES = {
    "citizen": {"label": "Người dân", "scopes": ("public",), "admin": False},
    "officer": {"label": "Cán bộ, công chức", "scopes": ("public", "internal"), "admin": False},
    "admin": {"label": "Quản trị tri thức", "scopes": ("public", "internal"), "admin": True},
}


class AuthError(ValueError):
    """Đăng nhập / token không hợp lệ."""


@dataclass(frozen=True)
class UserContext:
    user_id: str = "anonymous"
    role: str = "citizen"

    @property
    def scopes(self) -> tuple[str, ...]:
        return ROLES[self.role]["scopes"]

    @property
    def is_admin(self) -> bool:
        return ROLES[self.role]["admin"]

    @property
    def role_label(self) -> str:
        return ROLES[self.role]["label"]


ANONYMOUS = UserContext()


def _secret() -> bytes:
    secret = os.getenv("APP_SECRET", "")
    if not secret:
        # Không có APP_SECRET: khóa ngẫu nhiên theo tiến trình → token mất hiệu lực khi khởi động lại (an toàn hơn
        # một khóa cố định ghi trong mã nguồn)
        secret = _secret.fallback = getattr(_secret, "fallback", None) or secrets.token_hex(32)
    return secret.encode()


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, rounds, salt, digest = stored.split("$")
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds))
    return hmac.compare_digest(candidate.hex(), digest)


def _load_users(path: Path = USERS_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_user(username: str, role: str, password: str, path: Path = USERS_PATH) -> None:
    if role not in ROLES or role == "citizen":
        raise AuthError("Vai trò tài khoản phải là officer hoặc admin")
    if len(password) < 8:
        raise AuthError("Mật khẩu tối thiểu 8 ký tự")
    users = _load_users(path)
    users[username] = {"role": role, "password": hash_password(password)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def authenticate(username: str, password: str, path: Path = USERS_PATH) -> UserContext:
    user = _load_users(path).get(username)
    # So sánh băm cả khi không có tài khoản để thời gian phản hồi không lộ tên đăng nhập tồn tại hay không
    stored = user["password"] if user else hash_password("x")
    if not verify_password(password, stored) or not user:
        raise AuthError("Sai tên đăng nhập hoặc mật khẩu")
    return UserContext(user_id=username, role=user["role"])


def issue_token(user: UserContext, ttl: int = TOKEN_TTL_SECONDS) -> tuple[str, int]:
    expires = int(time.time()) + ttl
    body = base64.urlsafe_b64encode(json.dumps({"sub": user.user_id, "role": user.role, "exp": expires}).encode())
    signature = hmac.new(_secret(), body, hashlib.sha256).hexdigest()
    return f"{body.decode()}.{signature}", expires


def verify_token(token: str) -> UserContext:
    try:
        body, signature = token.rsplit(".", 1)
    except ValueError:
        raise AuthError("Token không hợp lệ") from None
    expected = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise AuthError("Token không hợp lệ")
    claims = json.loads(base64.urlsafe_b64decode(body.encode()))
    if claims["exp"] < time.time():
        raise AuthError("Phiên đăng nhập đã hết hạn")
    if claims["role"] not in ROLES:
        raise AuthError("Vai trò không hợp lệ")
    return UserContext(user_id=claims["sub"], role=claims["role"])


def _cli() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "list":
        for name, info in _load_users().items():
            print(f"{name:20} {info['role']}")
        return
    if len(sys.argv) != 4 or sys.argv[1] != "add-user":
        print(__doc__)
        sys.exit(1)
    username, role = sys.argv[2], sys.argv[3]
    password = os.getenv("NEW_USER_PASSWORD") or getpass.getpass(f"Mật khẩu cho {username}: ")
    save_user(username, role, password)
    print(f"Đã lưu tài khoản {username} ({ROLES[role]['label']}) vào {USERS_PATH}")


if __name__ == "__main__":
    _cli()
