<div align="center">

# VectorChord TileMaxSim

**基于 PostgreSQL 的向量/张量检索、精确 TileMaxSim 与常驻 GPU 缓存。**

[English](README.md) | 简体中文

</div>

VectorChord TileMaxSim 是一个构建在 PostgreSQL 上的开源检索引擎，代码源自
上游 [VectorChord](https://github.com/supervc-stack/VectorChord)。项目保留传统
单向量检索通路，并增加面向“每个文档由一组 token 向量表示”的精确
late-interaction 检索。

本仓库作为独立项目维护。README 只介绍本项目的实现、实测结果、局限和路线图，
不再附带上游官方 README，也不会把上游项目的宣传、镜像或产品指标表述成本项目
自己的能力。

## 项目定位

VectorChord TileMaxSim 提供基础设施能力：

- 在 PostgreSQL 中存储向量、张量描述符和检索元数据；
- 兼容传统单向量检索，并提供精确多向量 TileMaxSim；
- 让检索遵循 PostgreSQL 权限、MVCC、行可见性、取消和超时语义；
- 提供常驻 GPU 张量 arena、host 内存缓存和不可变磁盘分片；
- 为可选 GPU 服务提供有界准入、公平/优先级调度、缓存配额、健康探针和指标。

本项目不负责应用身份、认证、ACL 策略、实体注册表、Fact、事件账本、关系图谱、
社区或查询意图路由。这些属于经过认证的应用层或知识治理层。VectorChord 只接收
上层已经完成授权的候选范围，以及不具有授权含义的调度提示。

## 核心能力

- CPU 精确 TileMaxSim，以及原生 Rust/CUDA `tilemaxsimd` 后端。
- 不人为限制调用方授权范围内的张量候选数量，超出显存的工作集按有界批次处理。
- 外部张量源注册：PostgreSQL 保存稳定公开 ID 和描述符，完整 token 张量可存放在
  被注册的行外存储中。
- GPU/内存/磁盘三级张量路径：
  - L0：常驻 GPU 页面缓存；
  - L1：有界 host 内存缓存；
  - L2：持久存储上的不可变张量分片或内容寻址对象。
- 批量分片读取、批量 Host→Device 传输、内容寻址去重、TinyLFU/GDSF 准入，以及
  可合并的 page-run 分配器。
- 可选 resident 预热；预热完成前 daemon 不会报告 ready。
- daemon 内的公平、严格 priority 或 fair-priority 调度，请求 aging、租户权重、
  deadline、断连取消和有界 CUDA quantum。
- 多向量查询的 PostgreSQL 规划器统计与成本估算。
- Prometheus 指标，以及相互独立的存活和就绪探针。
- 正确性、注册表、协议、规划器、真实 GPU 和故障路径测试。

TileMaxSim 是可选功能。传统单向量通路不会启动 CUDA daemon，也不要求配置显存。
启用时，`tilemaxsimd` 会在启动阶段一次性预占用户指定设备上的 GiB 级显存；任何
设备或容量无法获得都会直接退出。

## 上层应用调用契约

NeoClaw/GBrain 等上层应用可以在一个事务内选择检索路径，连接池不会继承
`SET LOCAL` 的请求参数：

```sql
BEGIN;
SET LOCAL vchordrq.maxsim_backend = 'gpu';
SET LOCAL vchordrq.maxsim_scoring_profile = 'exact_fp16';
-- PQ/OPQ/RPQ 还需设置已激活的 qtc1 contract：
-- SET LOCAL vchordrq.maxsim_quantization_contract = 'qtc1-...';
SET LOCAL vchordrq.maxsim_tenant = 'scheduler-domain';
SET LOCAL vchordrq.maxsim_priority = 20;
-- 调用 vchordrq_tilemaxsim_search 或 vchordrq_tilemaxsim_rerank
COMMIT;
```

`maxsim_backend` 决定运行位置，`maxsim_scoring_profile` 独立决定张量表示和打分
kernel。可选 profile 为 `exact_fp16`、`int8`、`fp8_e4m3`、`pq` 和
`opq_rpq`。原生 daemon 已启用全部这些 profile。PQ family 使用带 SHA-256 身份校验的
VCTQ codebook/rotation 和内容寻址 VCTC codes；query rotation/LUT 构建后，ADC、
residual stage 累加和文档 token max 在原生 CUDA kernel 中融合。未激活、错配或损坏的
contract 会明确报错。系统绝不
静默改用另一精度。默认 backend 仍为 `coarse_only`，所以未显式启用 TileMaxSim
的应用不需要 GPU。
生产发布流程先用 `inspect_quantizer` 将训练所得 codebook digest 绑定进 schema-v2
contract ID，再原子写入 VCTQ/VCTC，并在 registry stage 前调用 `finalize_artifact`。
registry 和 daemon 会分别复核 artifact tree、内嵌 contract digest、codebook digest、
shape 与逐文件 checksum；激活和回滚使用带 generation 的一致快照。

候选范围与打分精度是另外两个独立决策：GBrain 可以传入结构化关系范围、授权范围
或普通语义查询的全 source 范围；VectorChord 不会把 scheduler tenant 当成权限，
也不会用量化粗排暗中裁掉调用方给出的候选。

## 架构

```text
认证后的应用 / 检索规划器
            │ 已授权 ID + 调度提示
            ▼
      PostgreSQL + VectorChord
            │ 张量描述符
            ▼
        tilemaxsimd
  ┌─────────┼──────────┐
  │         │          │
L0 GPU    L1 内存    L2 持久存储
缓存       缓存       分片
  │
  └── CUDA 精确 TileMaxSim
```

PostgreSQL 是行可见性和硬过滤的权威来源。daemon 不会根据 tenant 或 priority 推断
权限，只会对调用方明确传入的张量集合进行缓存解析和精确打分。

## GPU 缓存与调度

默认 `fair-priority` 调度器组合显式优先级和加权公平性。大请求按候选数、文档
token 数和实际 `query rows × document rows × dimension` 计算量切分。每个 CUDA
launch 结束后，请求重新进入调度器，因此其他调度域可以插入执行。已经开始的
CUDA kernel 不能被中断，FMA quantum 用来限制这段协作式不可抢占时间。

候选 quantum 默认为自动模式：daemon 选择所有已配置加速器都支持的最大有效校准
候选桶。用户传入非零 `--scheduler-quantum-candidates` 时仍作为硬覆盖；无法完成校准
时自动模式保守回退到 1024 个候选。

普通优先级的新一轮 busy period 最多等待 `--scheduler-batch-window-ms`，并在每个
quantum 边界连续吸收不超过 `--scheduler-max-microbatch-requests` 个兼容请求。
高优先级请求和已经恢复执行的长请求不会为了凑批额外等待。兼容条件包含租户、
priority、模型/量化 contract、shape，以及由
`--scheduler-min-shared-candidates-milli` 控制的内容寻址候选重叠率。候选完全一致、
采用精确 FP16 且全部命中 L0 的批次会进入共享候选 GPU 路径；其他情况确定性保持
原执行路径。

每张 GPU 启动时会覆盖 64/512/4096 候选桶、统一/中度分组/高度碎片化的文档行数
分布，以及 64--2048 个批内 query rows，对共享内存 tile kernel 与 cuBLAS FP16
Tensor Core GEMM + 分段 MaxSim 归约做有界 AB/BA 校准。只有数值一致且收益达到门槛
才采纳交叉点。grouped GEMM 的 candidate chunk 也只在用户配置的 workspace 内从
可用档位实测选择，不借用系统中未预留的空闲显存。执行层以匹配候选数，以及最接近
的 row-group 数、最大文档 rows 和 rows 不均衡度校准桶的交叉点积工作量为基准，直接比较实际
`query rows × document rows × dimension`；连续批处理带来的 query 复用和长尾文档
形状是两个独立输入。校准失败、workspace 不足、数值不一致或后端失败时确定性回退
tile/warp。请求/容量类失败只抑制对应 workload 桶，按 60、120、240 秒指数退避，
上限 30 分钟；连续三次设备类失败才打开全设备熔断，初始 30 秒、指数增长到最多
10 分钟。冷却结束进入 half-open，只放行恢复探测；Tensor 成功后关闭并重置对应退避。
状态与 `/metrics` 会暴露每个桶的阈值、实测
时间、chunk 大小、分类回退计数、抑制桶和熔断状态。兼容字段
`adaptive_tensor_threshold_rows` 只是所有桶中的最小阈值，不能再被当作运行时全局
分派阈值。
日志只记录限频后的 closed/open/half-open 状态转换；被抑制的日志事件仍通过累计
Prometheus counter 可见，不依赖日志做故障统计。

GPU 和 host cache 都有调度域最大占用限制。可选的
`--tenant-cache-reservation TENANT=GB` 会保护已经预热的 GPU 页面，使其他调度域
不能把它淘汰到低于指定容量。不过它当前只是“聚合字节保底”，不是某个知识库的
永久显存分区：它不会自动预热，不为 L1 内存提供最低保留量，也不能区分同一租户
拥有的多个 source。这些缺口已明确列入展望。

所有内存参数使用 GiB，不使用裸字节数。示例：

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
  --quantization-registry-root /srv/vectorchord/quantization-registry \
  --scheduler-policy fair-priority \
  --max-queued-requests 128 \
  --max-tenant-queued-requests 16 \
  --scheduler-quantum-fmas 4000000000 \
  --scheduler-max-microbatch-requests 8 \
  --scheduler-min-shared-candidates-milli 500 \
  --tenant-weight foreground=2 \
  --tenant-cache-reservation foreground=4
```

PostgreSQL 可以通过本地 Unix socket 或 `tcp://HOST:PORT` 连接 daemon。
`GET /livez`、`GET /healthz` 和 `GET /metrics` 分别提供存活、就绪和有界运行指标。
管理面还提供版本化视图：`GET /v1/config` 返回实际生效的 GPU、缓存、准入和调度
配置，`GET /v1/cache` 返回带版本与生成时间的 L0/L1 占用、碎片、命中、换入换出及
每张卡的 kernel 校准状态。运行时可通过 `POST /v1/cache/prewarm`、`/v1/cache/pin`
和 `/v1/cache/unpin` 管理内容寻址张量；`POST /v1/reload` 重载 shard 索引（量化激活状态
由每个请求原子读取）；
`POST /v1/devices/{ordinal}/tensor-circuit/probe` 主动触发 half-open 探测。异步操作
返回 operation ID，可通过 `GET /v1/operations/{id}` 查询状态。`POST /v1/drain`
会撤销 readiness、拒绝新请求和管理写、排空已接收工作后退出。所有写操作只允许通过
受文件权限保护的 Unix status socket；TCP status 端口始终只读。

上层应用应控制候选范围、backend、scoring profile、量化 contract、调度域、priority
和 deadline。显存预算、租户 reservation、准入上限、攒批阈值和 quantum 大小仍由
运维在实例启动时配置。系统不会暴露任意 eviction 或强制 kernel 接口，以免绕过缓存
隔离与校准回退。运行时 prewarm 遵守正常准入限制；只有启动期 resident manifest
是运维人员显式配置的强制 pinned 预热。

仓库保留容器构建定义，但不再自动发布镜像。人工发布时必须扫描最终镜像的不可变
digest，并在对外提供前附带 SPDX SBOM、provenance 和签名。
部署和协议细节见
[`docs/TILEMAXSIM_CUDA_SIDECAR.md`](docs/TILEMAXSIM_CUDA_SIDECAR.md) 与
[`docs/TILEMAXSIM_IPC_V2.md`](docs/TILEMAXSIM_IPC_V2.md)。

## 性能实测

27 组真实 GPU 量化、池化、PQ/RPQ/OPQ、融合算子和小显存消融的完整结果见
[《TileMaxSim 量化、池化与融合算子消融实验报告》](docs/tilemaxsim-quantization-results.zh-CN.md)。
多硬件 backend 边界、CPU/CUDA 隔离构建、NVIDIA 架构策略，以及 Metal、昇腾、
沐曦后端的验收门槛见
[《Portable accelerator backends》](docs/PORTABLE_ACCELERATOR_BACKENDS.md)。
首次参与开发可先阅读[《代码地图》](docs/CODEBASE_GUIDE.zh-CN.md)，其中按请求链路说明
PostgreSQL 扩展、daemon、三级缓存、调度、后端和测试的修改落点。
控制面 endpoint、请求格式、状态与错误语义见
[《TileMaxSim management API》](docs/TILEMAXSIM_MANAGEMENT_API.md)。
报告只公开聚合指标和可复现程序，不公开私有评测语料、生成张量、模型权重或内部路径。

开发机语料包含 34,054 个张量描述符、34,027 个唯一张量，逻辑 FP16 数据量为
16.28 GB。绝对延迟会随存储、CPU 和 GPU 改变，因此同机对照比孤立数字更有意义。

| 优化项 | 基线 | 优化后 | 结果 |
| --- | ---: | ---: | ---: |
| 不可变分片与批量读取 | 333.38 ms，顺序文件 | 56.52 ms | 快 5.90 倍 |
| 分片相对批量旧文件 | 87.56 ms | 56.52 ms | 快 1.55 倍 |
| 批量 Host→Device 传输 | 37.06 ms，100 次传输 | 14.19 ms，1 次传输 | 快 2.61 倍 |
| TinyLFU/GDSF 准入 | LRU 命中率 69.98% | 命中率 76.18% | +6.20 个百分点 |
| Rust/CUDA 冷请求 | Python/Triton 855.34 ms | 93.86 ms | 快 9.11 倍 |
| Rust/CUDA 热请求 p50 | Python/Triton 14.26 ms | 2.09 ms | 快 6.82 倍 |
| Rust/CUDA 热请求 p95 | Python/Triton 14.84 ms | 2.20 ms | 快 6.75 倍 |

20 GiB 实验将 18 GiB 分给张量、2 GiB 分给工作区。Rust/CUDA daemon 预热全部
34,027 个唯一张量耗时 14.86 秒，Python/Triton 实现为 23.07 秒。默认 32 KiB
page-run 分配器将原 256 KiB buddy allocator 的 17.840 GB 分配量降至
16.725 GB，内部取整浪费从 8.82% 降至 2.74%。

### 全范围与小缓存压力

全部张量常驻时，刻意对 34,054 个描述符做全量精确扫描平均为 1.08 秒。3 GiB
张量 arena 加 1 GiB host cache 返回相同的精确 top-K，但连续全量扫描产生
68,106 次 miss、61,499 次 eviction 和 32.53 GB Host→Device 传输；原生
round trip 平均达到 18.47 秒。

该差距来自循环缓存抖动和重复 I/O，不是显存缩小几倍后 TileMaxSim 算术就线性
变慢。高召回范围内的热工作集能放进缓存时，不会出现这种全库扫描行为。

### 候选范围消融

40 个真实查询使用原始 1,024 维文本 embedding/pgvector HNSW，与词法范围内的
精确 TileMaxSim 对照：

| 路径 | 平均延迟 | p95 | 质量 |
| --- | ---: | ---: | ---: |
| 原始文本 embedding + pgvector HNSW | 5.80 ms | 5.86 ms | 文档 hit@1 1.000 |
| 18 GiB 常驻缓存、候选范围内 TileMaxSim | 45.56 ms | 134.99 ms | hit@1 0.475；hit@5 0.650 |
| 3 GiB 缓存、候选范围内 TileMaxSim | 573.43 ms | 1,812.70 ms | TileMaxSim 排序相同 |

词法范围只覆盖了 65% 查询的相关文档；对于已经覆盖的查询，TileMaxSim 全部把相关
文档保留在 top 5。因此召回上限由候选生成而不是精确打分决定。普通词法和图谱命中
不能作为一般语义查询的唯一硬门；授权过滤仍然必须硬执行，明确的关系类意图可以在
上层能构造完整关系范围时使用图谱范围。

复现实验入口：

- [`services/benchmark_tilemaxsim_ablation.py`](services/benchmark_tilemaxsim_ablation.py)
- [`services/benchmark_full_corpus_tilemaxsim.py`](services/benchmark_full_corpus_tilemaxsim.py)
- [`services/benchmark_gbrain_scoped_tilemaxsim.py`](services/benchmark_gbrain_scoped_tilemaxsim.py)
- [`services/benchmark_postgres_single_vector.py`](services/benchmark_postgres_single_vector.py)

## 已知局限

本项目适合受控生产试点。严格低延迟、高可用或大规模多用户场景在正式 GA 前，必须
解决或明确接受以下限制。

### 检索规模

- 精确 TileMaxSim 是打分器，不是张量原生 ANN。全部常驻只能消除传输，不能消除
  精确计算量。
- 大规模低延迟知识库仍需要经过实测的高召回候选生成器，或未来的张量原生 ANN，
  再执行精确 TileMaxSim。
- 缓存小于活跃工作集时结果仍然正确，但可能发生抖动；生产容量必须覆盖热工作集，
  或保留经过验证的高召回范围。

### 缓存与多用户隔离

- 公平、priority、准入和 reservation 由每个 daemon 独立执行。多个副本之间没有
  全局队列、全局 entitlement 账本或感知缓存驻留的路由器。
- GPU reservation 保护的是某个调度 tenant 的聚合字节数，不是命名的
  `source_id` 或知识库；当前没有动态 cache-domain API、source 级预热契约或
  host-cache 最低保留量。
- reservation 在 daemon 启动时静态配置。多 GPU daemon 当前会把该数值应用到
  每张卡，而不是一个语义明确的集群总量。
- 全局 resident manifest 使用服务内部域固定。如果 pinned 页面占满 arena，
  无关的冷张量不能淘汰它们，请求会失败，而不是等待 pinned 数据离开。
- 抢占只能发生在 CUDA launch 之间，正在执行的 kernel 不可中断。
- tenant 和 priority 只是调度提示，不是授权证据。

### 运维与安全

- TileMaxSim TCP 协议没有内置 TLS 和客户端认证，必须放在私网或双向认证代理后。
- GPU 副本替换后是冷缓存。持久存储保证正确性，但可用性和热缓存延迟是两个独立
  SLO。
- PostgreSQL WAL 归档、PITR、跨站灾备和真实恢复验证仍属于数据库平台职责。
- 已提供指标，但还没有开箱即用的 SLO dashboard 和告警规则。
- 超大张量注册表与垃圾回收需要按实际规模做运维压测。

### 发布状态

SQL 接口、外部张量协议、镜像 tag 和部署封装仍在持续演进。真实 GPU 和故障路径
测试不能替代目标硬件上的长时间、多用户 soak/chaos 测试。

## 展望：AI Infra 的检索平面

VectorChord TileMaxSim 的目标是在可组合 AI Infra 中承担检索平面，而不是把应用
治理和模型服务职责都塞进自身：

```text
Agent / 应用层
      │
      ▼
AI Gateway：身份、配额、priority、deadline、Trace
      │
      ├── 知识层 ── PostgreSQL / VectorChord ── tilemaxsimd
      │
      └── Ray Serve ── vLLM 与其他模型引擎

Kubernetes / KubeRay：节点放置、GPU、扩缩容、故障恢复
对象存储：模型权重、张量分片、注入产物
Prometheus + OpenTelemetry：端到端可观测性
```

在这套分工中：

- [Ray Serve LLM](https://docs.ray.io/en/latest/serve/llm/) 负责分布式模型副本、
  自动扩缩容和模型感知路由；
- [vLLM](https://docs.vllm.ai/) 负责生成模型的连续批处理、模型并行、KV cache 和
  prefix cache；
- VectorChord 负责 PostgreSQL 检索语义以及向量/张量元数据，`tilemaxsimd` 负责
  精确 TileMaxSim、张量驻留和有界检索调度；
- 经过认证的应用层负责身份、ACL、图谱/Fact/事件治理、查询意图和最终 RAG 链路。

各组件不能在未感知对方的情况下竞争同一块显存。初期部署应让 vLLM 与 TileMaxSim
使用独立 GPU 池；Kubernetes 或其他资源管理器必须知道常驻 TileMaxSim daemon
已经预占的显存，之后再评估 MIG 或其他硬件分区方案。

下一阶段基础设施里程碑：

1. 将 `scheduler_tenant` 与不透明的 `cache_domain` 分离；后者通常由已认证上层根据
   tenant 和 source 派生，但 VectorChord 不解释其中的业务含义。
2. 为每个 cache domain 增加 GPU 和 host 的 `min_resident_gb`、`max_gb` 与预热
   契约，并明确区分单卡和全局语义。
3. 建立有界的全局准入层和感知张量驻留的多副本路由，同时保留 daemon 内精确的
   quantum 调度。
4. 让 request ID、tenant、source、priority、deadline、trace ID 和 cache domain
   组成统一请求信封，贯穿检索与模型服务。
5. 验证冷启动恢复、滚动升级、PostgreSQL 切换、GPU Pod 丢失、多租户隔离和长时间
   soak/chaos 行为。
6. 为无法用全范围精确 TileMaxSim 满足延迟 SLO 的大知识库增加张量原生 ANN，或
   其他经过召回率验证的第一阶段。

Ray 是编排层，不是持久存储。PostgreSQL 和不可变对象存储仍是权威数据源。
`tilemaxsimd` 应继续作为受监督的有状态 GPU 服务，而不是短生命周期的普通 Ray
任务，这样才能保证预占 arena 和缓存生命周期可预测。

## 安全边界

只有设置 `vchordrq.maxsim_tenant` 时，PostgreSQL 才发送带调度信息的协议。
priority 只改变延迟顺序，不能绕过行可见性、ACL、source 或其他硬过滤。日志只
暴露稳定哈希，不输出原始 tenant 标识。

本公开仓库只包含实现和对外项目信息，私有应用文档与规划文档不会提交到本仓库。

## 构建与测试

常用入口：

```shell
make build
cargo test --locked --workspace --exclude vchord --no-fail-fast
cargo test --manifest-path services/tilemaxsimd/Cargo.toml --locked
```

CUDA 容器及测试阶段定义在
[`services/Dockerfile.tilemaxsimd`](services/Dockerfile.tilemaxsimd)，PostgreSQL
张量源集成见 [`docs/MAXSIM_TENSOR_SOURCES.md`](docs/MAXSIM_TENSOR_SOURCES.md)。

## 致谢

感谢上游 [supervc-stack/VectorChord](https://github.com/supervc-stack/VectorChord)
项目的维护者与贡献者。其 PostgreSQL 扩展、向量类型、`vchordrq` 索引和已有
MaxSim 能力构成了本项目继续开发的基础。

本项目的 I/O 感知 GPU MaxSim 方向和 `TileMaxSim` 名称受到 Ashutosh Sharma
论文
[《TileMaxSim: IO-Aware GPU MaxSim Scoring with Dimension Tiling and Fused Product Quantization》](https://arxiv.org/abs/2606.26439)
（arXiv:2606.26439，2026）及其配套开源
[ashutoshuiuc/tilemaxsim](https://github.com/ashutoshuiuc/tilemaxsim) Triton 实现
的启发。感谢该工作公开的融合、分块 MaxSim 设计与性能分析。

本仓库是独立的 PostgreSQL/Rust/CUDA 系统集成，IPC 协议、GPU/内存/磁盘三级
张量缓存、分配器、多用户调度、数据库绑定和部署运维由本项目维护。上述致谢不表示
上游项目或论文作者对本项目背书，也不表示双方共同维护。

截至本文更新时，所引用的 TileMaxSim 仓库未发布许可证文件。因此，在未获得明确
授权前，不得将其源代码复制或改写进本仓库；未来任何获授权的复用也必须保留适用的
版权和许可声明。本仓库不直接包含该上游 Triton 仓库的源文件。

## 项目来源与许可

本项目源自 [supervc-stack/VectorChord](https://github.com/supervc-stack/VectorChord)，
原始源文件保留其上游版权和许可声明。本 fork 中独立创作的新增部分 Copyright
(c) 2026 Hu Xinjing。

仓库按 [`LICENSE`](LICENSE) 中的许可条款提供，在适用范围内包括 AGPLv3 和
Elastic License v2 选项。项目问题与支持请求请提交到本项目的
[Issue Tracker](https://github.com/HuXinjing/VectorChord/issues)。
