import time

import pytest

from src import auth


def test_password_hash_roundtrip():
    stored = auth.hash_password("matkhau-an-toan")
    assert stored.startswith("pbkdf2_sha256$") and "matkhau" not in stored
    assert auth.verify_password("matkhau-an-toan", stored)
    assert not auth.verify_password("sai", stored)


def test_users_file_and_roles(tmp_path):
    users = tmp_path / "users.json"
    auth.save_user("canbo01", "officer", "matkhau-123", path=users)
    user = auth.authenticate("canbo01", "matkhau-123", path=users)
    assert user.role == "officer" and user.scopes == ("public", "internal") and not user.is_admin
    with pytest.raises(auth.AuthError):
        auth.authenticate("canbo01", "sai-mat-khau", path=users)
    with pytest.raises(auth.AuthError):
        auth.authenticate("khong-ton-tai", "matkhau-123", path=users)
    with pytest.raises(auth.AuthError):
        auth.save_user("x", "citizen", "matkhau-123", path=users)      # người dân không cần tài khoản
    assert auth.ANONYMOUS.scopes == ("public",)


def test_tokens_are_signed_and_expire(monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    token, expires = auth.issue_token(auth.UserContext("admin01", "admin"))
    user = auth.verify_token(token)
    assert user.is_admin and expires > time.time()
    body, signature = token.rsplit(".", 1)
    with pytest.raises(auth.AuthError):
        auth.verify_token(f"{body}.{'0' * len(signature)}")              # sửa chữ ký
    forged_body = body[:-2] + ("AA" if body[-2:] != "AA" else "BB")
    with pytest.raises(auth.AuthError):
        auth.verify_token(f"{forged_body}.{signature}")                  # sửa nội dung (vd. nâng quyền)
    expired, _ = auth.issue_token(auth.UserContext("x", "officer"), ttl=-1)
    with pytest.raises(auth.AuthError, match="hết hạn"):
        auth.verify_token(expired)
