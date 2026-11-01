from datetime import timedelta

import pytest

from app.core.security import (
    TokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hash_roundtrip() -> None:
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong password", hashed)


def test_passwords_longer_than_bcrypt_limit_are_supported() -> None:
    password = "p" * 200
    hashed = hash_password(password)
    assert verify_password(password, hashed)
    assert not verify_password("p" * 199, hashed)


def test_access_token_roundtrip() -> None:
    payload = decode_access_token(create_access_token("user-123"))
    assert payload["sub"] == "user-123"
    assert payload["type"] == "access"


def test_expired_token_is_rejected() -> None:
    token = create_access_token("user-123", expires_delta=timedelta(seconds=-5))
    with pytest.raises(TokenError):
        decode_access_token(token)


def test_tampered_token_is_rejected() -> None:
    token = create_access_token("user-123")
    tampered = token[:-2] + ("aa" if token[-2:] != "aa" else "bb")
    with pytest.raises(TokenError):
        decode_access_token(tampered)


def test_garbage_token_is_rejected() -> None:
    with pytest.raises(TokenError):
        decode_access_token("not-a-jwt")


def test_verify_password_tolerates_invalid_hash() -> None:
    assert not verify_password("password", "not-a-bcrypt-hash")
