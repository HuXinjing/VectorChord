# VectorChord TileMaxSim 代码地图

本文面向第一次进入仓库的开发者，说明各模块职责、请求路径和修改落点。项目由两条
相互独立但可组合的路径组成：上游兼容的 PostgreSQL/VectorChord 索引路径，以及可选的
外部张量 TileMaxSim 加速路径。未配置 TileMaxSim 时，前者可以独立工作。

## 端到端数据流

```text
上层应用 / GBrain
  ├─ SQL：单向量、过滤条件、vchordrq 候选
  │    └─ PostgreSQL 扩展（src/ + crates/）
  └─ TileMaxSim 请求：候选 ID、张量描述符、profile、priority、deadline
       └─ PostgreSQL backend/IPC → tilemaxsimd
            ├─ 调度与连续微批
            ├─ L0 GPU / L1 host / L2 immutable shard
            ├─ CPU、CUDA 或独立厂商 backend
            └─ exact / INT8 / FP8 / PQ / OPQ-RPQ kernel
```

上层负责身份、ACL、知识结构和候选范围；VectorChord 负责向量/张量存储、检索、精确
打分和资源调度。请求中的 tenant 是公平性和缓存配额的调度域，不是授权边界。

## PostgreSQL 扩展

- `src/`：pgrx 扩展入口、SQL 绑定、数据类型、索引接入和运行时配置。
- `crates/vchordrq/`：IVF/RaBitQ 查询索引、候选生成、预取和 MaxSim 规划接口。
- `crates/vchordg/`：图索引路径。
- `crates/vector/`、`crates/distance/`、`crates/simd/`：向量表示、距离函数和 CPU SIMD。
- `crates/index/`、`crates/index_accessor/`：通用索引结构和 PostgreSQL 访问适配。
- `crates/rabitq/`、`crates/k_means/`：量化和聚类基础能力。
- `sql/install/`、`sql/upgrade/`：安装与升级 SQL；修改持久格式或 SQL contract 时必须同步
  增加升级、回滚和兼容测试。

## TileMaxSim daemon

`services/tilemaxsimd/` 是独立 Rust workspace，可按后端单独编译：

- `protocol.rs`：稳定 IPC frame、请求描述符、scoring profile 和错误响应。
- `daemon.rs`：socket、准入、reader 生命周期、调度主循环、管理 API、readiness/drain 和
  Prometheus 指标。
- `scheduler.rs`：fair-priority/strict-priority、aging、租户公平和服务额度。
- `dispatch.rs`：根据设备、query rows、候选数、文档长度和工作量选择 warp/tile/tensor
  路径；架构表只是校准失败时的保守后备。
- `engine.rs`：L0 命中、L1/L2 resolve、上传、批量打分、profile 编码和失败回滚的总编排。
- `cache.rs`：GPU page-run allocator、TinyLFU/GDSF 风格准入、租户 reservation、pin 和
  引用生命周期。
- `shard.rs`：不可变内容寻址 shard 与 host cache。
- `quant.rs`：版本化量化 contract、PQ code/codebook 的认证、激活与回滚。
- `backend.rs`、`backend_sdk.rs`：厂商无关 trait、ABI/能力协商和活体一致性探针。
- `gpu.rs`：CUDA kernel、cuBLAS/tensor 路径、校准与指数退避熔断。
- `cpu.rs`：CPU reference/SIMD 后端，也是其他硬件实现的正确性基准。
- `metal.rs`：Metal 接缝；其他 NPU 后端应通过 SDK 边界独立链接，不应塞进 CUDA 镜像。
- `bin/tilemaxsimctl.rs`：健康探测、对象发布和运维命令。

## 缓存和运行时管理

L2 shard 是正确性来源，L1 host cache 降低磁盘读取，L0 accelerator cache 避免重复 H2D。
运行时管理写只接受 Unix status socket：

- `/v1/cache/prewarm`、`pin`、`unpin`：异步缓存操作；普通运行时预热不能绕过准入。
- `/v1/reload`：重载 shard 索引；量化 registry 激活状态由请求路径原子读取。
- `/v1/devices/{ordinal}/tensor-circuit/probe`：人工请求熔断 half-open 探测。
- `/v1/operations/{id}`：查询有界操作账本。
- `/v1/drain`：撤销 readiness，拒绝新工作，排空已接收请求并退出。

TCP status 只读。`/v1/config` 用于控制面能力发现，`/v1/cache` 是带版本和时间戳的状态
快照，`/metrics` 是长期监控接口。

## 构建、部署与测试

- `services/Dockerfile.tilemaxsimd`：NVIDIA CUDA 镜像；`-blackwell` 是对应架构构建。
- `services/Dockerfile.tilemaxsimd-cpu`：无 GPU 的 reference/fallback 镜像。
- `services/Dockerfile.postgres`：包含扩展的 PostgreSQL 镜像。
- `deploy/systemd/`：非容器部署示例。
- `tests/`：PostgreSQL 扩展回归；`services/test_tilemaxsim_rust_daemon.py`：真实 daemon/IPC
  集成；`services/tilemaxsimd` 内 Rust tests：allocator、调度、协议、量化与 kernel contract。
- `services/Dockerfile.*` 与验证脚本：当前采用人工构建和发布；发布者必须对最终镜像
  digest 执行扫描，并附带 SBOM、provenance 和签名。

常见修改落点：调度策略改 `scheduler.rs`；kernel 选择改 `dispatch.rs`；CUDA 实现改
`gpu.rs`；缓存策略改 `cache.rs`，但跨层上传/回滚同时检查 `engine.rs`；协议字段先改
`protocol.rs`，再同步数据库调用端、版本协商和兼容测试。

## 必须保持的边界

- TileMaxSim 为可选能力；未显式配置 accelerator memory 时不应改变普通 VectorChord 路径。
- ACL 在可信上层和数据库硬过滤中执行，调度 tenant 不可替代授权。
- 粗筛、量化与缓存只能改变性能，不能静默改变所声明 scoring profile 的语义。
- 所有异步状态必须有上限、可查询、可观测，并在错误或 drain 时释放 lease/reference。
- 不同硬件共享协议和 conformance contract，但使用独立构建产物；不要为硬件拆长期功能分支。
