"""Streamlit登录组件；会话令牌不进URL、共享缓存或日志。"""

import streamlit as st

from .api_client import AgentApiClient, AgentApiError


def get_session_client():
    """只在本浏览器会话复用连接池；显式空令牌禁止回退到.env共享令牌。"""
    if "http_client" not in st.session_state:
        st.session_state["http_client"] = AgentApiClient(
            token=st.session_state.get("access_token", "")
        )
    return st.session_state["http_client"]


def clear_login():
    client = st.session_state.get("http_client")
    if client:
        client.close()
    st.session_state.clear()  # 清掉旧用户消息、候选记忆、身份和客户端。
    st.query_params.clear()  # 不能把前一账号的thread继续带给后一账号。


def describe_login_error(error):
    if error.is_connection_error:
        return (
            "无法连接后端，请先启动后端并等待Application startup complete，再点击重试。"
        )
    if error.status_code == 429:
        return "尝试过于频繁，请5分钟后再试。"
    return error.message  # 后端账号接口只返回脱敏的稳定错误信息。


def render_login(client):
    """登录前只探测公开接口，不请求聊天、知识库信息和用户数据。"""
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("⚖️ 劳动法知识库")
        st.caption("有据可查 · 明细可核 · 会话独立")
        st.markdown("登录后使用法条问答、加班费工具和个人会话记忆。")
        try:
            config = client.auth_config()
        except AgentApiError as error:
            st.error(describe_login_error(error))
            if st.button("重试连接", key="login_retry"):
                st.rerun()
            return
        if config["mode"] != "accounts":
            st.warning(
                "后端处于旧令牌模式。请设置AGENT_AUTH_MODE=accounts并重启后端后使用账号登录。"
            )
            return
        login_tab, register_tab = st.tabs(["账号登录", "注册账号"])
        with login_tab:
            with st.form("login_form", clear_on_submit=True):
                username = st.text_input("用户名", max_chars=32, key="login_username")
                password = st.text_input(
                    "密码", type="password", max_chars=128, key="login_password"
                )
                submitted = st.form_submit_button(
                    "登录", type="primary", use_container_width=True
                )
            if submitted:
                try:
                    result = client.login(username, password)
                    authenticated = AgentApiClient(
                        base_url=client.base_url, token=result["access_token"]
                    )
                    try:
                        identity = (
                            authenticated.me()
                        )  # 身份仍由后端返回，不信表单自报。
                    except AgentApiError:
                        authenticated.close()
                        raise
                    client.close()
                    st.session_state.clear()
                    st.query_params.clear()
                    st.session_state.update(
                        {
                            "access_token": result["access_token"],
                            "http_client": authenticated,
                            "identity": {**identity, "username": result["username"]},
                        }
                    )
                    st.rerun()
                except AgentApiError as error:
                    st.error(describe_login_error(error))
        with register_tab:
            if not config["registration_enabled"]:
                st.info("管理员已关闭自助注册。")
            else:
                st.caption("用户名3—32位，以字母开头，允许字母、数字、下划线和连字符。")
                with st.form("register_form", clear_on_submit=True):
                    new_username = st.text_input(
                        "设置用户名", max_chars=32, key="register_username"
                    )
                    new_password = st.text_input(
                        "设置密码",
                        type="password",
                        max_chars=128,
                        key="register_password",
                    )
                    confirmation = st.text_input(
                        "再次输入密码",
                        type="password",
                        max_chars=128,
                        key="register_confirmation",
                    )
                    agreed = st.checkbox(
                        "仅创建本项目专用密码；不填写公司机密或敏感个人资料。"
                    )
                    register = st.form_submit_button(
                        "创建账号", use_container_width=True
                    )
                if register:
                    if not agreed:
                        st.warning("请先确认使用提示。")
                    elif new_password != confirmation:
                        st.error("两次输入的密码不一致。")
                    else:
                        try:
                            client.register(new_username, new_password)
                            st.success("注册成功，请切换到“账号登录”后登录。")
                        except AgentApiError as error:
                            st.error(describe_login_error(error))
        st.divider()
        st.caption(
            "本地学习演示系统，不构成法律意见。刷新或断开浏览器会话后可能需要重新登录。"
        )


def logout():
    try:
        get_session_client().logout()
    except AgentApiError as error:
        if not error.is_unauthorized:
            st.error("无法确认后端已注销，请恢复连接后重试；当前登录仍保留。")
            return
    clear_login()
    # 由按钮回调触发的Streamlit重跑会自动显示登录页。
