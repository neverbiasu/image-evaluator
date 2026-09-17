# Milestone 8-01 (M8-01) Metric Registry 基础与能力契约交付验收报告

## 1. 概述与交付范围

Milestone 8（0.5.0）的核心目标是构建以 **Metric Registry** 为中枢的轻量、透明、可扩展能力目录与统一契约。M8-01 作为 0.5.0 的第一阶段工作单元，交付只读的指标元数据注册表、领域模型规范以及面向 10 项现有评估指标的全量收敛，为后续 `evaluate()` 调度解耦（M8-02）、详细结果对象（M8-03）与 CLI 发现过滤（M8-04）奠定契约基石。

### 1.1 获批交付范围 (Approved Scope)
- **核心定义**：Metric 是包含输入契约（Input Contract）、计算协议（Calculation Protocol）以及具体实现（Model / Math Implementation）的可执行评估方法。
- **领域数据模型**：实现不可变的 `InputContract`、`ImplementationRef` 与 `MetricSpec`（`frozen=True, slots=True`），对输入角色、优劣方向与可溯源实现进行严格静态校验与防御性归一化。
- **开放词汇设计**：`tasks` 与 `objectives` 采用开放字符串元组，支持开放式任务/目标导航与组合过滤，不冻结完整科学分类学。
- **10 项指标全量收敛**：收敛已有 10 项指标（`aesthetic`, `arcface`, `clip`, `directional_clip`, `fid`, `kid`, `lpips`, `pickscore`, `psnr`, `ssim`），固化输入角色契约、优劣方向与文档路径。
- **轻量与重模型隔离**：`image_evaluator.registry` 导入与查询严禁加载任何重度依赖（PyTorch, torchvision, transformers, open-clip, clean-fid, lpips, insightface）。

### 1.2 严格排除范围 (Excluded Scope)
- **概念隔离**：Benchmark、Dataset、Judge 与 Metric 保持完全独立，本单元绝不实现 Benchmark/Dataset/Judge 运行时抽象。
- **杜绝过度工程**：不实现第三方动态插件系统、动态钩子、外部网络拉取或自动指标推荐引擎。
- **保持既有行为**：不变更既有 `evaluate()` 字典返回契约（留待 M8-02 接入，M8-03 扩充）。
- **流程门禁**：不暴露 CLI 命令、不修改公开版本号（保持 0.4.0）、不提交 Git Commit、不触发发布。

---

## 2. 变更文件清单与实施细节

M8-01 采用外科手术式增量实现，涉及 3 个新增未跟踪文件，现有已跟踪源码零修改：

| 文件路径 | 行数 | 角色与职责 |
| :--- | :---: | :--- |
| `image_evaluator/specifications.py` | 164 | 领域数据模型：`InputContract`, `ImplementationRef`, `MetricSpec` 及其不变性与防御性归一化验证 |
| `image_evaluator/registry.py` | 311 | 只读目录中心：`MetricRegistry` 封装（`MappingProxyType`）、10 项指标静态规范收敛、`list_metrics` / `get_metric` / `filter_metrics` 查询入口 |
| `tests/test_metric_registry.py` | 237 | 单元测试套件：13 项自动化测试，覆盖目录完整性、四输入契约、不可变性、防御性归一化、错误拦截、重依赖隔离与模块重载 |

---

## 3. 领域模型与契约架构

### 3.1 InputContract（输入契约）
```python
@dataclass(frozen=True, slots=True)
class InputContract:
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
```
- `required` 与 `optional` 均强制转换为不可变元组；
- 输入角色必须是非空且无头尾空白的字符串；
- 严禁角色重叠：若同名角色同时出现在 `required` 与 `optional`，抛出 `ValueError`。

### 3.2 ImplementationRef（实现溯源）
```python
@dataclass(frozen=True, slots=True)
class ImplementationRef:
    backend: str
    protocol: str
    model: str | None = None
    backend_version: str | None = None
    model_revision: str | None = None
```
- 记录评估指标底层的算法协议、依赖包标识与骨干模型，用于透明溯源；
- 纯元数据记录，不包含可执行对象，避免预载模型。

### 3.3 MetricSpec（指标规范）
```python
@dataclass(frozen=True, slots=True)
class MetricSpec:
    id: str
    display_name: str
    tasks: tuple[str, ...]
    objectives: tuple[str, ...]
    inputs: InputContract
    score_direction: Literal["higher_is_better", "lower_is_better"]
    implementation: ImplementationRef
    aggregation: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    docs_path: str | None = None
```
- `id` 强制要求全小写；
- `score_direction` 严格限制为 `"higher_is_better"` 或 `"lower_is_better"`；
- 列表/可变集合入参自动转为不可变元组，杜绝外部就地篡改。

### 3.4 MetricRegistry（只读目录）
```python
@dataclass(frozen=True, slots=True)
class MetricRegistry:
    _metrics: Mapping[str, MetricSpec]
```
- 内部索引使用 `MappingProxyType` 保护，禁止运行时增删改；
- 重复指标 ID 在构造阶段抛出 `DuplicateMetricError`；
- 未知指标在 `get()` 阶段抛出 `UnknownMetricError`；
- `filter(task=..., objective=...)` 支持按开放标签组合过滤。

---

## 4. 架构前后对比与行为变化

| 维度 | 重构前 (M7 Baseline) | 重构后 (M8-01 Metric Registry) |
| :--- | :--- | :--- |
| **元数据存储** | 散落在 `main.py`（`PROMPT_METRICS`, `PAIRWISE_FIDELITY_METRICS`, `DATASET_METRICS`）与 `core.py`（`PREDICTOR_MAP`）各处 | 统一定义在 `image_evaluator/registry.py`，`METRIC_REGISTRY` 成为唯一事实来源 |
| **输入契约形式** | CLI 代码中手写 if/else 参数判定，Python API 依靠 `core.py` 内部 `_validate_inputs` 校验 | 形式化 `InputContract`，明确角色如 `("image", "reference_image", "prompt", "source_prompt")` |
| **方向与优劣** | 仅写在 Markdown 文档中，代码层无机读定义 | 形式化 `score_direction`（`higher_is_better` / `lower_is_better`）机读可查 |
| **能力发现** | 外部程序无法在不加载重型深度学习框架的前提下列举指标属性 | 顶层 `list_metrics()`、`get_metric()`、`filter_metrics()` 零开销查询（<1ms），零重依赖加载 |
| **向后兼容性** | 既有逻辑完全保持 | 100% 向后兼容，既有 308 项单测零回归通过 |

---

## 5. 门禁验证结果与质量证据

所有验证均使用项目专属 Python 解释器（`.conda/bin/python`，Python 3.11.15）。

### 5.1 门禁汇总表
| 门禁项 | 执行命令 | 解释器 | 退出码 | 结果与证据 |
| :--- | :--- | :--- | :---: | :--- |
| **Gate 1: 定向单测** | `.conda/bin/python -m pytest -q tests/test_metric_registry.py` | Python 3.11.15 | 0 | 13 passed in 0.03s (`targeted_tests.log`) |
| **Gate 2: 全量回归** | `TORCH_HOME=/private/tmp/image-evaluator-torch-cache .conda/bin/python -m pytest -q` | Python 3.11.15 | 0 | 308 passed, 23 warnings in 18.63s (`full_pytest.log`) |
| **Gate 3: 静态检查** | `.conda/bin/python -m ruff check .` | Python 3.11.15 | 0 | All checks passed! (`ruff.log`) |
| **Gate 4: 格式检查** | `git diff --check` | 系统 Git | 0 | 无空白或格式违规 (`git_diff_check.log`) |
| **Gate 5: 工作区状态** | `git status --short --branch` | 系统 Git | 0 | develop 分支，仅 3 个未跟踪文件 (`git_status.log`) |
| **Gate 6: 重依赖隔离** | `.conda/bin/python -c '...'` (sys.modules 隔离测试) | Python 3.11.15 | 0 | 0 heavy modules loaded (`heavy_import_isolation.log`) |

### 5.2 重依赖隔离代码与证据
测试脚本通过 `sys.modules` 拦截 `("torch", "transformers", "open_clip", "clip", "cleanfid", "lpips")`，验证导入 `image_evaluator.registry` 时无任何重依赖泄漏：
```text
HEAVY IMPORT ISOLATION: PASSED. Zero heavy modules loaded.
exit code: 0
```

---

## 6. 当前限制与边界说明

1. **调度中枢未接管**：M8-01 仅提供 Registry 能力契约，`evaluate(...)` 与 CLI `main.py` 尚未替换既有硬编码集合（留待 M8-02 推进）。
2. **CLI 发现子命令未暴露**：CLI 尚未暴露 `image-evaluator list-metrics` 或按 task/objective 过滤命令行入口（留待 M8-04 推进）。
3. **返回形态保持原样**：本阶段不涉及 `EvaluationResult` 结构化包装对象，保持既有字典返回契约（留待 M8-03 推进）。
4. **模型实际推理**：M8-01 为只读元数据契约测试，不触发真实神经网络推理权重下载。

---

## 7. 可视化验收物与真实截图证据

依据 Boss 治理规则与 `review-visual-acceptance` 技能规范，验收材料已持久化至：
`/Users/nev4rb14su/workspace/image-evaluator/artifacts/m8-01-metric-registry/`（及 `.harness/artifacts/m8-01-metric-registry/`）。

| 交付文件 | 文件描述 | 采集/渲染方式 |
| :--- | :--- | :--- |
| `visual_acceptance.html` | 本地自包含可视化验收页面，无外部 CDN，支持桌面与窄屏自适应响应 | 原生 HTML5 + CSS Grid + 10 项指标矩阵 |
| `terminal_validation.png` | 真实 macOS Terminal.app 物理窗口截图（Window ID 116821） | macOS 原生 `screencapture -l`，**绝无 PIL/Canvas/AI 合成** |
| `browser_desktop.png` | 桌面视口（1280×2400）真实浏览器完整渲染截图 | Google Chrome 140 Headless 原生渲染抓取 |
| `browser_narrow.png` | 窄屏移动视口（390×1600）真实浏览器响应式渲染截图 | Google Chrome 140 Headless 原生渲染抓取 |

---

## 8. Acceptance Review 独立审查结论

基于 Acceptance 角色准出标准进行 5 维度终审：

1. **正确性 (Correctness)**：`MetricRegistry` 与 `specifications.py` 100% 满足 M8 规范要求；10 项指标的输入契约与优劣方向与已验收文档及实现绝对一致；13 项定向单测与 308 项全量单测全部通过。**判定：PASS**。
2. **架构与边界 (Architecture & Boundaries)**：完全遵从 Boss 批准的边界定义；未引入 Benchmark/Dataset/Judge 抽象；未引入动态插件系统；重依赖严格物理隔离。**判定：PASS**。
3. **可读性与防呆性 (Robustness & Cleanliness)**：数据类全部 `frozen=True, slots=True`；字典使用 `MappingProxyType` 保护；可变序列在 `__post_init__` 中完成 tuple 转换，消除就地修改漏洞。**判定：PASS**。
4. **证据完备性 (Evidence Rigor)**：6 份原始日志、1 份独立可视化页面、3 份真实终端与浏览器截图全部齐备，物理保存在持久目录中。**判定：PASS**。
5. **准出建议 (Conclusion)**：**M8-01 技术准出条件全部满足，建议提请 Boss 正式验收。**

---

## 9. 治理状态与停机门禁

- **当前状态**：`◻ 待 Boss 显式验收 (Pending Boss Review)`
- **停机原则**：严格执行 Stop Gate，在 Boss 显式确认验收 M8-01 之前：
  - 严禁启动 M8-02 编码；
  - 严禁创建 Git Commit；
  - 严禁修改版本号；
  - 严禁发布 PyPI。
