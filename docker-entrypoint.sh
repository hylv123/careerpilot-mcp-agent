#!/bin/sh
# ============================================================
# 容器启动脚本（entrypoint）
# ============================================================
# 它解决的问题：容器启动前要先做点准备工作，做完再把接力棒交给真正的服务。
#
# 为什么不能用 CMD 直接写多条命令：
#   CMD 只能指定一个主进程。而"建索引 + 起服务"是两件事，
#   混在一起会出现信号传不过去的问题（见文件末尾 exec 的说明）。

set -eu   # -e：任何一步失败立刻退出；-u：用了没定义的变量就报错，避免拼错变量名还静默通过

# ------------------------------------------------------------
# 启动自检：把缺配置这件事说在前面
# ------------------------------------------------------------
# 为什么需要：
#   这些变量缺失时，服务本身照样起得来，只有真正调 /analyze 时才会报错，
#   而且报的是笼统的 502，你得翻半天日志才知道是配置没配齐。
#   在启动时就把缺的东西喊出来，能把排查时间从"十分钟"降到"十秒钟"。
#
# 为什么只警告不退出：
#   健康检查、API 文档这些不依赖密钥的部分仍然可用，
#   让容器活着、信息明确，比让它起不来更好排查。
missing=0
for name in LLM_API_KEY LLM_BASE_URL LLM_MODEL DASHSCOPE_API_KEY EMBEDDING_MODEL; do
    if [ -z "$(printenv "$name" 2>/dev/null || true)" ]; then
        echo "[entrypoint] ⚠️ 缺少环境变量：$name"
        missing=1
    fi
done
if [ "$missing" = "1" ]; then
    echo "[entrypoint]    请对照 .env.example 补齐上面的变量，否则 /analyze 与检索会失败"
fi

INDEX_FILE="/app/knowledge/index/index.faiss"

need_build=0
if [ "${FORCE_REBUILD_INDEX:-0}" = "1" ]; then
    # 你在 knowledge/docs/ 里换了新资料，想让索引重新生成
    echo "[entrypoint] 检测到 FORCE_REBUILD_INDEX=1，强制重建索引"
    need_build=1
elif [ ! -f "$INDEX_FILE" ]; then
    # 第一次启动，还没有索引
    echo "[entrypoint] 未检测到向量索引，准备构建"
    need_build=1
else
    echo "[entrypoint] 已存在向量索引，跳过构建"
fi

if [ "$need_build" = "1" ]; then
    if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
        # 为什么这里只警告而不是直接退出：
        #   密钥缺失不代表服务完全废了——健康检查、接口文档、其它节点都还能用，
        #   只有"经历检索"这一步会报错。让它启动起来，用户看到明确的错误信息
        #   比看到一个起不来的容器更容易排查。
        echo "[entrypoint] ⚠️ 未设置 DASHSCOPE_API_KEY，跳过索引构建；检索功能将不可用"
    else
        if python -m knowledge.build_index; then
            echo "[entrypoint] 索引构建完成"
        else
            echo "[entrypoint] ⚠️ 索引构建失败，服务继续启动，但检索功能会报错"
        fi
    fi
fi

# ------------------------------------------------------------
# 关键的一行：exec "$@"
# ------------------------------------------------------------
# exec 会用后面的命令【替换掉】当前这个 shell 进程。
# 为什么要替换：
#   容器停止时，Docker 会给"1 号进程"发 SIGTERM 让它体面退出。
#   如果这里写成 `python -m uvicorn ...`（不带 exec），PID 1 是这个 shell，
#   shell 收到信号后不一定会转发给子进程，结果就是 uvicorn 收不到信号，
#   只能等 Docker 超时后强杀（SIGKILL）——正在进行的工作会直接断掉。
#   exec 之后 uvicorn 自己就是 PID 1，信号直达，可以优雅关闭。
#
# "$@" = 传进来的所有参数，在 compose/Dockerfile 的 CMD 里指定。
exec "$@"
