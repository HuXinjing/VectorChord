<div align="center">

# VectorChord TileMaxSim

**面向 PostgreSQL 精确多向量检索的开源 VectorChord 分支。**

[English](README.md) | 简体中文

</div>

本分支为 VectorChord 的 `vchordrq` 索引增加精确 late-interaction
TileMaxSim 检索，适用于“每个文档保存一个 token 向量张量，并在
PostgreSQL 内完成多向量搜索”的应用。

VectorChord 在这里负责向量/张量存储、精确 TileMaxSim、GPU 缓存和调度。
实体、事件、Fact、关系图谱、社区、意图路由、用户身份与 ACL 等应用治理能力
应由 GBrain 等上层系统负责，不进入 VectorChord。

## 本分支增加的能力

- CPU 精确 TileMaxSim 重排，以及可选的原生 Rust/CUDA 后端。
- 不人为限制调用方授权范围内的张量候选数量；显存压力由 GPU 页面缓存、
  分块计算和有界请求调度处理。
- 外部张量源注册，可将完整精度的 token 张量存放在 PostgreSQL 行外。
- 外部张量查询路径遵循 PostgreSQL 权限、MVCC、行可见性、取消和超时语义。
- 多向量查询的规划器统计与成本估算。
- GPU/内存/磁盘三级存储路径、常驻预热、缓存换入换出和指标接口。
- 按 daemon 实例执行的公平调度、优先级、准入限制和缓存配额。
- 正确性、注册表、sidecar 协议、规划成本和真实 GPU 故障路径测试。

TileMaxSim 是可选功能。没有显式配置 GPU 显存时不会启动 CUDA daemon，
传统单向量 VectorChord 使用不需要 GPU。启用时，daemon 会在启动阶段预占用户
指定设备上的 GiB 级显存；任何设备或容量无法获得都会直接失败退出。

## 性能实验

开发机语料包含 34,054 个张量描述符、34,027 个唯一张量，逻辑 FP16 数据量
为 16.28 GB。绝对延迟会随存储、CPU 和 GPU 改变，因此同机对照比单个数字
更有参考价值。

| 优化项 | 基线 | 优化后 | 结果 |
| --- | ---: | ---: | ---: |
| 不可变分片与批量读取 | 333.38 ms，顺序文件 | 56.52 ms | 快 5.90 倍 |
| 分片相对批量旧文件 | 87.56 ms | 56.52 ms | 快 1.55 倍 |
| 批量 Host→Device 传输 | 37.06 ms，100 次传输 | 14.19 ms，1 次传输 | 快 2.61 倍 |
| TinyLFU/GDSF 准入 | LRU 命中率 69.98% | 命中率 76.18% | +6.20 个百分点 |
| Rust/CUDA 冷请求 | Python/Triton 855.34 ms | 93.86 ms | 快 9.11 倍 |
| Rust/CUDA 热请求 p50 | Python/Triton 14.26 ms | 2.09 ms | 快 6.82 倍 |
| Rust/CUDA 热请求 p95 | Python/Triton 14.84 ms | 2.20 ms | 快 6.75 倍 |

20 GiB 显存实验把 18 GiB 用于张量、2 GiB 用于 TileMaxSim 工作区，服务就绪前
预热全部 34,027 个唯一张量。Python/Triton 从进程启动到就绪耗时 23.07 秒，
Rust/CUDA daemon 为 14.86 秒；预热后的常驻请求不再从磁盘读取张量。

GPU 缓存从一块 CUDA arena 中分配连续 page run，并使用 best-fit 尺寸桶和按
地址合并。默认 32 KiB 页面将语料的分配空间从原 256 KiB buddy allocator 的
17.840 GB 降至 16.725 GB，内部取整浪费从 8.82% 降至 2.74%。页面大小可以用
`--gpu-block-kib` 调整。

### 全范围检索与小缓存压力

18 GiB 张量 arena 中全部张量常驻时，34,054 个描述符的刻意全量精确扫描平均
为 1.08 秒。3 GiB 张量 arena 加 1 GiB host cache 返回相同的精确 top-K，但
连续全量扫描产生 68,106 次 miss、61,499 次 eviction 和 32.53 GB Host→Device
传输，平均延迟达到 18.47 秒。

这说明小缓存下结果仍然正确，但循环抖动会显著增加延迟；它不是“显存缩小几倍，
GPU 算术就线性变慢”。高召回范围内的热工作集能够放入缓存时不会出现这种全库
扫描行为。

在 40 个真实查询上的传统 `text-embedding-v4`/pgvector HNSW 对照如下：

| 路径 | 平均延迟 | p95 | 质量 |
| --- | ---: | ---: | ---: |
| 原始文本 embedding + pgvector HNSW | 5.80 ms | 5.86 ms | 文档 hit@1 1.000 |
| 18 GiB 常驻缓存、候选范围内精确 TileMaxSim | 45.56 ms | 134.99 ms | 文档 hit@1 0.475；hit@5 0.650 |
| 3 GiB 缓存、候选范围内精确 TileMaxSim | 573.43 ms | 1,812.70 ms | 排序相同 |

该实验中的词法范围召回率只有 0.650，因此词法或普通图谱结果不能作为一般语义
检索的唯一硬门。它们可以加权、解释或预热缓存；关系类意图可以使用完整图谱范围，
而 source、ACL、文档类型等授权约束仍必须硬过滤。

复现实验入口：

- [`services/benchmark_tilemaxsim_ablation.py`](services/benchmark_tilemaxsim_ablation.py)
- [`services/benchmark_full_corpus_tilemaxsim.py`](services/benchmark_full_corpus_tilemaxsim.py)
- [`services/benchmark_gbrain_scoped_tilemaxsim.py`](services/benchmark_gbrain_scoped_tilemaxsim.py)
- [`services/benchmark_postgres_single_vector.py`](services/benchmark_postgres_single_vector.py)

## GPU 缓存与调度

默认 `fair-priority` 调度器组合显式优先级和加权公平性。大请求会按候选/token
计算量切分，并在 CUDA kernel 之间重新进入调度器。队列、进行中请求内存、GPU
和 host cache 都可以设置全局及调度域上限。

内存配置使用 GiB，不使用字节数。示例：

```shell
tilemaxsimd \
  --socket /run/vectorchord/tilemaxsim.sock \
  --listen 0.0.0.0:9191 \
  --status-socket /run/vectorchord/tilemaxsim-status.sock \
  --gpu-memory-gb 0=20 \
  --gpu-workspace-gb 2 \
  --host-cache-gb 8 \
  --max-inflight-request-gb 1 \
  --contract-root MODEL_CONTRACT_ID=/srv/vectorchord/tensors \
  --scheduler-policy fair-priority \
  --max-queued-requests 128 \
  --max-tenant-queued-requests 16 \
  --scheduler-quantum-fmas 4000000000 \
  --tenant-weight foreground=2 \
  --tenant-cache-reservation foreground=4
```

发布镜像使用 CUDA 12.6 构建，需要 NVIDIA Container Runtime，以及兼容
CUDA 12.6 的宿主机驱动。最终镜像保留静态链接的 CUDA 应用运行时，但不包含
不需要的完整 CUDA toolkit；驱动库和 GPU 设备由 NVIDIA 运行时注入。

PostgreSQL 的 `vchordrq.maxsim_gpu_endpoint` 可以是本地 Unix socket，也可以是
`tcp://HOST:PORT`。`GET /livez`、`GET /healthz` 和 `GET /metrics` 分别用于
存活、就绪和 Prometheus 指标检查。

## 已知限制（Limitations）

本分支可以用于受控生产试点，但在严格低延迟、高可用或大规模多用户场景正式
GA 前，必须解决或明确接受以下限制。

### 检索规模

- TileMaxSim 是精确打分器，不是张量原生 ANN。source、ACL、文档类型等授权硬
  过滤可以安全缩小范围，但普通词法或图谱命中不能充当一般语义查询的高召回硬门。
- 张量全部常驻 GPU 只能消除存储传输，不能消除精确计算量。本次实测全量 34,054
  个描述符平均 1.08 秒，而高召回应用范围平均 45.56 ms。大知识库的严格低延迟
  部署仍需要张量原生 ANN，或经过召回率验证的候选生成器，再做精确 TileMaxSim。
- 缓存小于活跃工作集时不会返回错误结果，但可能频繁换入换出。前述 3 GiB 压力
  实验的全量扫描平均 18.47 秒，生产容量必须覆盖热工作集或保留高召回范围。

### 多用户与调度

- 公平、priority、准入上限和缓存配额目前由每个 daemon 独立执行。多个 GPU
  副本之间还没有全局队列、全局 entitlement 账本或感知工作量的负载均衡器，
  因此可能出现副本负载倾斜。
- 抢占发生在 CUDA kernel 之间，无法中断已经开始执行的 kernel；FMA quantum
  负责限制单次不可抢占的最长计算量。
- GPU 故障转移可依靠持久张量存储保持正确性，但新副本是冷缓存。高可用和热缓存
  延迟是两个独立 SLO，需要分别验证。
- tenant 标识和 priority 只是调度提示，不是授权凭据。VectorChord 不负责应用
  多租户、身份、ACL、图谱治理或特权名单；上层认证系统必须只传入已授权硬范围。

### 部署、安全与灾备

- TileMaxSim TCP 协议当前没有内置 TLS 和客户端认证，只能放在私网并通过防火墙
  或 Kubernetes NetworkPolicy 限制 PostgreSQL 调用；跨信任域时应增加双向认证
  代理。
- 容器、探针、指标、PostgreSQL 集成和适配 HA 的 TCP 服务已经提供，但 manifest
  不等于生产验证。必须在目标环境演练 PostgreSQL 主从切换、GPU Pod 丢失、滚动
  升级、备份恢复和数据迁移。
- 本仓库不负责持续 WAL 归档、PITR 和跨站灾备；这些属于数据库平台职责，而且
  必须做真实恢复测试。
- 已提供指标接口，但没有开箱即用的 SLO dashboard 和告警规则。生产环境需要监控
  排队、拒绝、超时、缓存抖动、磁盘容量、GPU 健康和故障后的冷缓存延迟。
- 张量对象数量很大时，GC 文件扫描和数据库锁可能变得可见，需要按实际规模压测并
  规划维护窗口或后续分代/墓碑式 GC。

### 发布状态

SQL 接口、外部张量协议、镜像 tag 和部署封装仍在持续开发，稳定版之前可能变化。
仓库中的真实 GPU 与故障路径测试验证了有界执行和正确性，但不能替代目标硬件上的
多小时、多用户 soak/chaos 压测。

## 安全边界

PostgreSQL 只有在设置 `vchordrq.maxsim_tenant` 时才发送 v3 调度元数据，否则
保持 v2 协议兼容。上层系统可以设置 -100 到 100 的 priority，但优先级只影响
延迟顺序，不能绕过 PostgreSQL 行可见性、ACL、source 或其他硬过滤。

本仓库只包含公开实现与公开项目信息，不包含私有规划或上层应用文档。

## 项目关系

本项目基于 [supervc-stack/VectorChord](https://github.com/supervc-stack/VectorChord)。
上游安装方式、SQL 用法、兼容性与许可证信息请参阅[英文 README](README.md) 中保留的
原始 VectorChord 文档。
