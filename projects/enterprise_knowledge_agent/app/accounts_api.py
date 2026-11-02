"""公开的注册/登录入口；只返回脱敏错误，不把密码放进422响应或日志。"""

import os
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .accounts import AccountError, AccountStore
from .auth import parse_bearer


router = APIRouter(prefix="/auth", tags=["account"])


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 不允许客户端在注册时附加user_id或角色。
    username: str = Field(min_length=3, max_length=32)
    password: SecretStr = Field(min_length=12, max_length=128)


class AccountIdentity(BaseModel):
    user_id: str
    username: str


class LoginResponse(AccountIdentity):
    access_token: str
    token_type: str
    expires_at: int


def get_account_store(request: Request) -> AccountStore:
    store = getattr(request.app.state, "account_store", None)
    if store is None:
        raise HTTPException(
            503,
            detail={
                "code": "accounts_unavailable",
                "message": "账号登录尚未就绪；请使用accounts模式启动后端。",
            },
        )
    return store


def registration_enabled():
    return os.getenv("AGENT_ALLOW_REGISTRATION", "true").strip().lower() == "true"


def account_http_error(error):
    return HTTPException(
        error.status,
        detail={"code": error.code, "message": error.message},
        headers={"Retry-After": "300"} if error.status == 429 else None,
    )


def peer_address(request):
    return request.client.host if request.client else "unknown"


@router.get("/config")
def auth_config(request: Request):
    mode = getattr(request.app.state, "auth_mode", "accounts")
    return {
        "mode": mode,
        "registration_enabled": mode == "accounts" and registration_enabled(),
    }


@router.post("/register", response_model=AccountIdentity, status_code=201)
def register_account(
    body: Credentials,
    request: Request,
    store: Annotated[AccountStore, Depends(get_account_store)],
):
    if not registration_enabled():
        raise HTTPException(
            403,
            detail={"code": "registration_disabled", "message": "管理员已关闭注册。"},
        )
    try:
        return store.register(
            body.username, body.password.get_secret_value(), peer_address(request)
        )
    except AccountError as error:
        raise account_http_error(error) from None


@router.post("/login", response_model=LoginResponse)
def login_account(
    body: Credentials,
    request: Request,
    store: Annotated[AccountStore, Depends(get_account_store)],
):
    try:
        return store.login(
            body.username, body.password.get_secret_value(), peer_address(request)
        )
    except AccountError as error:
        raise account_http_error(error) from None


@router.post("/logout")
def logout_account(
    store: Annotated[AccountStore, Depends(get_account_store)],
    authorization: Annotated[str | None, Header()] = None,
):
    token = parse_bearer(authorization)
    if store.authenticate(token) is None:
        raise HTTPException(
            401, detail={"code": "unauthorized", "message": "登录已失效。"}
        )
    store.logout(token)
    return {"logged_out": True}
