# syntax=docker/dockerfile:1
# ============================================================
# CareerPilot MCP Agent —— 镜像构建文件
# ============================================================
# 这份文件的目标：让别人拿到代码后，不用装 Python、不用配环境，
# 只用一条命令就能把你的服务跑起来。
#
# 阅读顺序建议：从上往下，每一行我都在旁边写了"为什么这么做"。


# ============================================================
# 第 1 步：选基础镜像
# ============================================================
# python:3.11-slim  =  官方 Python 3.11 + 精简版 Debian
#
# 为什么不用 python:3.11（完整版）：
#   多了几百 MB 的编译工具和文档，运行时根本用不上，白占体积和下载时间。
#
# 为什么不用 python:3.11-alpine（更小）：⭐️ 这是最容易踩的坑
#   alpine 用的是 musl libc，而 faiss-cpu / numpy 这类包发布的预编译包
#   基本都是针对 glibc（manylinux）的。alpine 上装它们会退化成"从源码编译"，
#   构建时间从 1 分钟变成十几分钟，还经常因为缺编译依赖直接失败。
#   "更小的镜像"省下的体积，远不如"能构建成功"重要。
#
# 为什么还要锁到 "3.11.10-slim-bookworm" 这么具体，而不是写 python:3.11-slim：
#   python:3.11-slim 是【浮动标签】，它今天指向 3.11.10，半年后可能指向 3.11.14。
#   也就是说同同一份 Dockerfile，在不同时间构建会得到不同的底层系统，
#   某天 CI 突然挂了你根本查不出原因。锁到"小版本 + 系统代号"才能保证可复现。
#   （这是刚踩出来的：写 python:3.11-slim 时拉取耗了 11 分钟也没成功，
#    而本地已经存在 3.11.10-slim-bookworm，直接复用即可。）
FROM python:3.11.10-slim-bookworm

# 元信息：别人拿到镜像 docker inspect 时能看到这是什么
LABEL org.opencontainers.image.title="CareerPilot MCP Agent"
LABEL org.opencontainers.image.description="输入岗位 JD，输出匹配报告、简历要点与面试题"


# ============================================================
# 第 2 步：设置运行时环境变量
# ============================================================
# 这里全是"写死进镜像"的环境变量，都是【非敏感】的运行时行为开关。
# 敏感的东西（API Key）绝对不能出现在这里，它们在运行时通过 compose 注入。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    CAREERPILOT_DB_PATH=/app/data/jobs.db \
    CAREERPILOT_OUTPUT_DIR=/app/output

# 逐个解释：
#   PYTHONUNBUFFERED=1
#       让 print / loguru 的输出【立刻】写出去，不要攒在缓冲区。
#       不设的话你在 `docker logs` 里可能半天看不到一行日志，
#       会误以为程序卡死了——实际上日志都憋在内存里。这是容器化最常见的困惑。
#
#   PYTHONDONTWRITEBYTECODE=1
#       不生成 .pyc 文件。容器是一次性的，生成了也没下次，纯占空间。
#
#   PYTHONIOENCODING / LANG / LC_ALL = utf-8
#       精简镜像的默认编码是 ASCII。你的 JD 和报告全是中文，
#       一旦输出中文就会 UnicodeEncodeError。这三个一起设才稳妥。
#
#   PIP_NO_CACHE_DIR=1
#       pip 默认会留安装缓存，镜像里没用，能省几十 MB。
#
#   CAREERPILOT_DB_PATH / CAREERPILOT_OUTPUT_DIR
#       程序本身支持从环境变量读这两个路径（见 mcp_servers/job_server.py）。
#       指向 /app/data 和 /app/output，这两个目录稍后会用数据卷挂载，
#       这样容器删了，你的数据和生成的文档还在。


# ============================================================
# 第 3 步：工作目录
# ============================================================
# 后面所有相对路径都基于它。用一个专用目录而不是根目录，
# 避免不小心覆盖了系统自带的东西（比如 /data 这种常见目录名）。
WORKDIR /app


# ============================================================
# 第 4 步：安装依赖（单独一层）
# ============================================================
# ⭐️ 这里是整个 Dockerfile 性能的关键：
#   Docker 是【分层】构建的，每一条指令算一层，并且会缓存。
#   如果某一层没变，下次构建直接复用缓存。
#
#   所以这里【只拷贝 requirements.txt】而不是 `COPY . .`：
#       依赖清单很少改动 → 这一层长期命中缓存
#       而你每天改几十次的业务代码在下一层 → 永远只用重新拷一遍、不用重装依赖
#
#   反过来，如果你写成 `COPY . . && pip install -r requirements.txt`，
#   那你改一个字的注释，都会导致 60 秒的依赖重装。
# ---------- 构建参数：pip 装包用哪个源 ----------
# 为什么用 ARG 而不是 ENV：
#   ENV 会把变量永久写进镜像环境，而这个源只有"装依赖"这一步需要，
#   运行期留着它属于不必要的残留。ARG 只在构建期存在。
#
# 为什么要留这个开关：
#   容器里 pip 默认走 pypi.org，国内网络经常超时。
#   在 docker-compose.yml 的 build.args 里换成一个国内镜像地址就能解决，
#   而在国外的同学（或 GitHub Actions）不传这个参数，会走默认官方源——两边都不耽误。
ARG PIP_INDEX_URL=https://pypi.org/simple

# 只拷 API 运行期依赖。界面层那一份（requirements-ui.txt）刻意不带进来：
#   Streamlit 会顺带拖进 pandas / numpy / altair / tornado / pillow 等一大串，
#   镜像白白胖一截，还多一堆用不上的攻击面。
COPY requirements.txt .

# --timeout / --retries：faiss-cpu 这类包有好几十 MB，
#   默认的 15 秒超时在弱网下一拉就断，给足时间并允许重试。
RUN python -m pip install --upgrade pip --timeout 60 --retries 5 \
    && python -m pip install --timeout 60 --retries 5 -r requirements.txt

# 为什么合并成一条 RUN（用 && 连起来）而不是写多条：
#   每条 RUN 都是一层。pip 升级和装依赖合成一层，
#   否则升级 pip 那层会产生独立的缓存垃圾，镜像白白变大几十 MB。


# ============================================================
# 第 5 步：拷贝业务代码
# ============================================================
# 放在依赖安装【之后】，就是为了上面说的缓存策略。
# .dockerignore 里过滤掉的（.env、.venv、tmp/ 等）在这一步被排除。
COPY . .


# ============================================================
# 第 6 步：创建非 root 用户
# ============================================================
# 为什么不能直接用 root 跑：
#   root 是容器内权限最高的账户。万一你的代码（或某个依赖）被植入恶意逻辑，
#   它能读写容器里的一切。用普通用户跑，能把损失限制在这一个用户能碰的范围。
#   这是"容器安全"里最基本、也最常被面试官问到的一条。
#
# 注意顺序：必须在 COPY 之后 chown，否则新拷进来的文件还是 root 的。
# ⚠️ /app/knowledge/index 必须在这里【显式创建】并把属主改成 appuser。
#   踩过一次的坑：.dockerignore 排除了 knowledge/index/，镜像里压根没这个目录；
#   Docker 挂载数据卷到一个不存在的路径时，会自己 mkdir —— 但用的是【root】去创建，
#   于是容器里以 appuser 运行的进程写不进去，索引构建直接
#       RuntimeError: could not open /app/knowledge/index/index.faiss: Permission denied
#   先把目录按正确属主建好，卷挂上来权限才是对的。data/output/logs 同理。
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/output /app/logs /app/knowledge/index \
    && chown -R appuser:appuser /app

# ---------- 入口脚本授权 ----------
# Windows / macOS 上写的 .sh 文件常常没有"可执行"权限位，
# 在 Dockerfile 里显式 chmod 一次，保证跨平台都能跑。
RUN chmod +x /app/docker-entrypoint.sh

ENV PATH="/home/appuser/.local/bin:${PATH}"

# 从这一行开始，后面的命令都以 appuser 身份执行
USER appuser


# ============================================================
# 第 7 步：端口与健康检查
# ============================================================
# EXPOSE 只是"声明"这个容器会监听 8000 端口（相当于文档），
# 真正把端口映射出来是在 docker-compose.yml 或 docker run -p 里做的。
EXPOSE 8000

# 健康检查：Docker 每隔一段时间跑一下这条命令，判断容器是否真的活着。
#
# 为什么用 python 而不是 curl：
#   slim 镜像里默认没有 curl（再装一个会多一层），
#   但 Python 是现成的，直接用标准库发个请求就行。
#
# start_period 很重要：
#   服务刚启动时要建索引、加载模型客户端，几十秒内起不来是正常的。
#   给它 60 秒宽限期，否则前几次健康检查失败会被判定为 unhealthy，
#   甚至被编排工具（K8s）直接杀掉重启。
HEALTHCHECK --interval=30s \
            --timeout=5s \
            --start-period=60s \
            --retries=3 \
            CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"


# ============================================================
# 第 8 步：启动命令
# ============================================================
# ENTRYPOINT 和 CMD 的分工：
#   ENTRYPOINT = 固定的"前缀"，永远是入口脚本（负责准备工作）
#   CMD        = 默认参数，传给入口脚本执行用户可以在 docker run 时覆盖
#
# 最终执行的是：docker-entrypoint.sh uvicorn api.main:app --host 0.0.0.0 ...
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# uvicorn 参数说明：
#   --host 0.0.0.0  ⭐️ 必须！
#       默认只监听 127.0.0.1（容器内部的回环地址），
#       那样容器外面根本连不进来，你会看到"端口映射没错但就是访问不上"。
#       这是新手容器化排错第一名。
#
#   --workers 1
#       不用多进程。原因有两：
#       1) 每个 worker 会各自初始化一份 MCP 工具和大模型客户端，内存翻倍；
#       2) 这个项目瓶颈在等模型响应（IO），不在 CPU 计算。
#       而 FastAPI 里那些 def 路由本来就在线程池里跑，并发能力已经有了。
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
