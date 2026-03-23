---
title: "在autodl上搭建Codex日志"
description: "无 GUI 远程 Linux 服务器搭通 Codex 的完整排查记录"
publishDate: 2026-03-23
category: technique
tags: ["codex", "linux", "remote-ssh", "autodl"]
draft: false
comment: true
---

## 问题背景

场景很明确：

- 本地主机已经可以正常登录 Codex。
- 远程服务器是无 GUI 的 Linux 环境（通过 VS Code Remote-SSH 使用，本文章是在autodl上的Linux环境）。
- 远程服务器直接登录 Codex (Sign in ChatGPT, 而不是使用API) 不顺利。
- 最终采用了“本地主机代理 + SSH 反向端口转发 + auth.json 迁移”这条路径打通。

下面只按实际排查顺序记录过程。

## 第一步：先确认本地主机代理端口

先在本地主机查常见端口是否监听。如果没查到，再去代理客户端界面看实际端口。

```bash
# 本地主机：检查常见代理端口监听
ss -lntp | grep -E ':(xxxx|7890|1080|3128)\b'
```

确认到本地主机可用的 HTTP 代理端口后，用 `curl` 直接验证是否能走到 OpenAI：

```bash
# 本地主机：验证代理链路是否可达 OpenAI
curl -x http://127.0.0.1:xxxx -I https://api.openai.com/v1/models
```

这一步如果返回 `401 Unauthorized`，判断为链路已通。

## 第二步：在本地主机发起 SSH 反向端口转发

这条命令必须在本地主机执行，不是在远程服务器执行。

```bash
# 本地主机执行：把本地代理端口 xxxx 反向映射到远程 yyyy
ssh -p xxxxx -N -R yyyy:127.0.0.1:xxxx xxx@xxx.example.com
```

执行后，远程服务器就可以通过 `127.0.0.1:yyyy` 访问本地主机代理。这个 SSH 窗口必须保持打开。

调试时踩过一个坑：曾经把 `ssh -R` 误敲到远程服务器里，结果当然不会生效。纠正后在本地主机重新执行才打通。

## 第三步：在远程服务器上配置代理环境变量

先在远程服务器导出 `HTTP_PROXY / HTTPS_PROXY`，随后发现环境里还有旧的小写变量和 `ALL_PROXY`，会干扰结果。因此改为先清理，再同时设置大小写版本。

```bash
# 远程服务器：清理旧代理变量
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# 远程服务器：统一设置为新映射端口
export http_proxy=http://127.0.0.1:yyyy
export https_proxy=http://127.0.0.1:yyyy
export HTTP_PROXY=http://127.0.0.1:yyyy
export HTTPS_PROXY=http://127.0.0.1:yyyy
```

用下面两条命令确认当前环境和实际出网：

```bash
env | grep -i proxy
curl -I https://api.openai.com/v1/models
```

如果这里返回 `401 Unauthorized`，说明远程默认出网已走到新代理链路。

## 第四步：处理远程 shell 自动覆盖代理的问题

这是关键问题之一：即使手动设置了新代理，重新登录 shell 后又被恢复成旧端口。

排查发现远程 shell 里存在 `watch_proxy` / `clashon` 一类自动脚本；当小写 `http_proxy` 为空时，它会自动切回旧值。

处理方法不是反复手动 `export`，而是把正确代理提前写进 shell 初始化文件，确保自动脚本运行前就拿到正确值：

```bash
# 远程服务器：写入 ~/.bashrc（或你实际使用的初始化文件）
export http_proxy=http://127.0.0.1:yyyy
export https_proxy=http://127.0.0.1:yyyy
export HTTP_PROXY=http://127.0.0.1:yyyy
export HTTPS_PROXY=http://127.0.0.1:yyyy
```

随后新开一个 shell，再次 `env | grep -i proxy` 验证，确认新环境能稳定生效。

## 第五步：迁移 Codex 的 auth.json

在本地主机确认 `~/.codex/auth.json` 已存在（前提是本地主机 Codex 已登录成功），然后复制到远程服务器同路径，并收紧权限。

```bash
# 本地主机：确认文件存在
ls -l ~/.codex/auth.json

# 本地主机执行：复制到远程服务器
scp -P xxxxx ~/.codex/auth.json xxx@xxx.example.com:~/.codex/auth.json

# 远程服务器：限制权限
chmod 600 ~/.codex/auth.json
```

这一步适用于“本地主机能登录，远程服务器直接登录困难”的情况。`auth.json` 是敏感文件，不能泄露。

## 第六步：重连 VS Code Remote-SSH 让环境生效

因为远程环境变量和 `auth.json` 都改过，必须让 VS Code 远端环境重新加载。

操作顺序：

1. 断开 Remote-SSH 连接。
2. 重新连接 `xxx.example.com`。
3. 必要时执行“重新加载窗口（Reload Window）”。

这里遇到过一个现象：VS Code 可能复用旧的远端 server / 扩展宿主，导致新环境没有立即生效。重连加重载后通常恢复。

## 第七步：登录成功后仍然出现 Reconnecting 的排查

现象是：已经能登录，但发送问题后持续显示 `Reconnecting`。这一段按固定顺序排查，不再重复改登录。

1. 先排除基础网络

在远程服务器持续跑连通性检查：

```bash
while true; do
  date
  curl -I --max-time 15 https://api.openai.com/v1/models >/dev/null && echo OK || echo FAIL
  sleep 5
done
```

连续 `OK` 说明基础代理链路稳定，问题不在“网络根本没通”。

2. 再看 Codex 与 Remote-SSH 日志（关键）

这里不是泛看日志，而是按下面的检查顺序：

   1. 打开 `View -> Output`，先看 `Codex`，再看 `Remote - SSH`。
   2. 把时间窗口收敛到 `Reconnecting` 出现前后 1-2 分钟。
   3. 在 Codex 里重点找：`fetch failed`、`timeout`、`child process`。
   4. 在 Remote-SSH 里重点找：`connection reset`、`disconnected`、`resolver error`。

当时观察到：

- Codex 有 `TypeError: fetch failed`（如 `/wham/usage`）和 `timeout waiting for child process to exit`。
- Remote-SSH 没有 SSH 主连接中断信号。
- Remote-SSH 反而出现 `Found running server`、`reusing cached exec server`。

据此判断：更像 Codex 扩展后端/子进程或扩展宿主复用旧状态，而不是 SSH 主链路故障。

3. 转向刷新远端 VS Code 状态

围绕上面的判断，后续动作改为刷新远端运行态：

   1. 断开并重连 Remote-SSH。
   2. 执行 `Developer: Reload Window`。
   3. 若仍异常，执行 `Kill VS Code Server on Host` 后再连接。

这一段的核心结论是：先用 `curl` 排除链路问题，再用两类日志做归因，最后处理远端扩展宿主/VS Code Server 的复用状态。

## 最后总结

这套方案核心只有两件事：

1. 通过 SSH 反向端口转发，让远程服务器借用本地主机代理。
2. 通过复制 `auth.json`，让远程服务器复用本地主机已登录的 Codex 状态。

后续继续使用时，通常要先在本地主机重新建立 SSH 反向隧道；`auth.json` 不一定每次都要复制，只有失效时再同步即可。

## 最小命令清单

```bash
# 1) 本地主机：确认代理端口可用
curl -x http://127.0.0.1:xxxx -I https://api.openai.com/v1/models

# 2) 本地主机：建立反向隧道（窗口保持打开）
ssh -p xxxxx -N -R yyyy:127.0.0.1:xxxx xxx@xxx.example.com

# 3) 远程服务器：清理并设置代理
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export http_proxy=http://127.0.0.1:yyyy
export https_proxy=http://127.0.0.1:yyyy
export HTTP_PROXY=http://127.0.0.1:yyyy
export HTTPS_PROXY=http://127.0.0.1:yyyy

# 4) 远程服务器：验证代理是否生效
env | grep -i proxy
curl -I https://api.openai.com/v1/models

# 5) 本地主机 -> 远程服务器：迁移 Codex 登录态
scp -P xxxxx ~/.codex/auth.json xxx@xxx.example.com:~/.codex/auth.json
ssh -p xxxxx xxx@xxx.example.com "chmod 600 ~/.codex/auth.json"
```
