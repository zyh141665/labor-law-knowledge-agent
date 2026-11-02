# 账号登录（本地学习演示版）

## 用户看到的界面

未登录：居中的“劳动法知识库”标题、能力说明，以及“账号登录/注册账号”两个标签。未登录不渲染问答、聊天记录和长期记忆。

注册：用户名、密码、再次输入密码、专用密码使用提示。注册成功后切换登录；没有默认管理员、默认密码或自动创建的测试账号。

登录后：原有问答界面与工具保持不变，侧边栏显示账号和“退出登录”。当前版本保留Streamlit原生组件，不添加第三方前端框架。

## 启动与验收

修改位于`D:\量化agent\labor-law-knowledge-agent`。已有向量索引和BGE文件无需重建或重新下载。

1. 在现有前后端CMD窗口各按Ctrl+C停止旧进程。
2. 后端窗口运行：

```cmd
cd /d D:\量化agent\labor-law-knowledge-agent
D:\量化agent\agent-learning\.venv\Scripts\activate.bat
python -m projects.enterprise_knowledge_agent.scripts.run_backend
```

3. 等待启动完成，另一个窗口运行：

```cmd
cd /d D:\量化agent\labor-law-knowledge-agent
D:\量化agent\agent-learning\.venv\Scripts\activate.bat
python -m projects.enterprise_knowledge_agent.scripts.run_frontend --port 8502
```

4. 打开`http://127.0.0.1:8502`，注册本项目专用账号再登录。用户名3—32位，以字母开头，允许字母、数字、下划线和连字符；不区分用户名大小写。密码12—128个字符，建议使用随机且不复用的密码。
5. 错误密码应失败；正确登录后显示问答；退出应回登录页，旧会话令牌再请求应401。
6. 使用无痕窗口注册第二个账号；第二个账号不应看到第一个账号的聊天/记忆。可先不发模型问题，只验证注册与页面切换，这些操作不调用模型。

如果出现`/auth/config`的404，通常仍是旧后端；检查终端是否从新项目启动，并确认8000端口的旧服务已停止。

## 配置

```dotenv
AGENT_AUTH_MODE=accounts
AGENT_ALLOW_REGISTRATION=true
```

两项未设置时，默认是上面的本地演示配置；不必修改现有`.env`即可体验。设置`AGENT_ALLOW_REGISTRATION=false`可以关闭后续自助注册。

`AGENT_AUTH_MODE=legacy_tokens`保留旧脚本的静态Bearer令牌方式，仍需有效`AGENT_API_TOKENS`；新版登录界面不在旧模式下自动使用共享令牌。accounts模式明确忽略旧令牌，不把它当作后门。

## 后端协议

| 接口 | 是否需登录 | 用途 |
|---|---|---|
| GET /auth/config | 否 | 登录方式和注册开关，不返回用户列表或密钥 |
| POST /auth/register | 否 | 接收username/password，只返回生成的user_id和用户名 |
| POST /auth/login | 否 | 返回随机访问令牌、8小时有效期和当前身份 |
| POST /auth/logout | 是 | 注销当前会话；不会退出该账号其他设备的会话 |
| GET /me | 是 | 由后端根据令牌确定user_id |

后续请求仍使用`Authorization: Bearer <登录返回的令牌>`。请求体/查询参数里的user_id必须与令牌身份匹配；修改URL或请求体不能冒充别人。注册请求禁止自行指定user_id或角色。

## 存储与安全取舍

- 新增`data/accounts.sqlite`，保存账号、带盐密码哈希、会话令牌摘要和认证尝试计数；不提交Git。
- 密码使用标准库PBKDF2-HMAC-SHA256、600000次迭代、每人独立随机盐；不是简单SHA256，也不是可逆加密。随机会话令牌具有高熵，仅存SHA256摘要，不能混同密码哈希用途。
- 每次登录签发独立令牌，8小时过期，最多保留20个活跃会话；退出删除当前令牌摘要。重启数据库仍保留有效登录，浏览器会话断开后没有自动“记住我”。
- 每个浏览器使用自己的`st.session_state`及HTTP客户端，不使用全局`st.cache_resource`缓存带身份的客户端；不把密码或令牌写进URL、日志、共享缓存或浏览器localStorage。
- 账号接口422响应不回显输入密码；账号响应设置`Cache-Control: no-store`。密码字段使用SecretStr减少意外repr泄露；这不等于密码传输加密。
- 五分钟内每个登录账号最多10次尝试、登录来源IP60次、注册来源IP10次，计数持久化；429返回Retry-After。统计所有尝试，包括成功登录。没有复杂验证码、风控或分布式防护。
- IP取实际连接，不信任客户端伪造的X-Forwarded-For。经Streamlit请求时多个用户可能共享前端IP，因此可能共同触发IP限制。
- 默认仅本机127.0.0.1。不得直接把演示配置暴露公网；公网需要HTTPS、注册策略、账号恢复、审计和更完整的防护。PBKDF2不是声称已达到FIPS合规。
- 登录失败使用相同的用户名/密码错误提示，不区分不存在的账号与错误密码；注册用户名重复仍会提示不可用。
- 退出不删聊天或长期记忆；只是吊销当前登录权限。账号数据库与业务数据库不是同一个事务，当前没有账号删除功能。

## 旧数据如何处理

原来的`student-001`等令牌身份不是新账号。新账号由后端生成`acct_随机编号`，故首次登录历史为空是预期，旧聊天和记忆未删除。不能通过注册同名用户接管旧数据；如需迁移，应另做确认归属后的管理操作，本次不自动迁移。

原有三个业务数据库继续各司其职：`checkpoints_v2.sqlite`、`chat_history.sqlite`、`memory.sqlite`；加上账号数据库后总计四个SQLite文件。

## 验证

测试使用临时数据库、HTTP测试客户端和Streamlit AppTest。覆盖密码哈希/盐、会话过期与注销、用户名冲突、限流、跨账号403、旧令牌拒绝、输入脱敏、登录页面、两个模拟浏览器隔离和退出清理。不创建真实用户、不调用付费模型。

```cmd
python -m pytest -q projects\enterprise_knowledge_agent\tests
```

参考：[Python PBKDF2接口](https://docs.python.org/3/library/hashlib.html#hashlib.pbkdf2_hmac)、[OWASP密码存储指南](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)。
