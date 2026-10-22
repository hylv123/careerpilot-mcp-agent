# Roadmap（后续可做的方向）

> 这里只是"如果继续做会做什么"，不代表当前实现。当前能力与边界见 [README 的「当前状态」与「已知限制」](../README.md)。

## 工程化

- [ ] Streamlit 端到端演示界面（现在只有命令行与 HTTP 接口）
- [ ] worker 心跳与任务级进度上报 —— 当前只能发现"进程没了"，发现不了"进程卡死但不退出"
- [ ] 可选的 Streamable HTTP MCP 部署（跨主机场景；现在是本地 stdio）
- [ ] 简历与面试材料 PDF 导出（现在只产 Markdown）

## 检索与评测

- [ ] 混合检索（BM25 + 向量）与可配置重排
- [ ] 用"召回片段命中多少 JD 核心技能词"替代距离阈值来判断是否触发查询改写
      —— 这是 Recall@k 评测暴露出的真实缺陷，见 [EVOLUTION.md](EVOLUTION.md)
- [ ] 评测补齐延迟、成本与多次运行的稳定性指标

## 已完成（记录用）

- [x] 离线单元测试与 GitHub Actions
- [x] Docker Compose 本地启动
- [x] 运行期依赖与界面层依赖拆分（`requirements-ui.txt` 不进 API 镜像）
- [x] Redis 异步任务队列（Stream + 消费组）、幂等去重、限流、Redis 不可用自动降级
