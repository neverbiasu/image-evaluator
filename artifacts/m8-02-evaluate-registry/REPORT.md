# M8-02 验收报告：evaluate() 注册中心对接与 Directional CLIP 接入

## 1. 基本信息
- **工作项**: M8-02 (`evaluate()` 注册中心对接与 Directional CLIP 接入)
- **里程碑**: Milestone 8 (0.5.0)
- **分支**: `develop`
- **当前版本**: `0.4.0` (严格遵循治理规则：未获 Boss 验收前不修改版本号，不提交 Git，不发布)
- **修改文件白名单**:
  - `image_evaluator/core.py`
  - `tests/test_api_evaluate.py`
- **交付证据目录**:
  - `artifacts/m8-02-evaluate-registry/`
  - `.harness/artifacts/m8-02-evaluate-registry/`

---

## 2. 变更背景与架构目标
在 M8-01 中，项目引入了集中式指标注册中心（`MetricRegistry` 与 `MetricSpec`），统一了指标元数据、入参依赖与适用场景。然而，主 API 入口 `image_evaluator.evaluate()` 此前仍维护着硬编码的集合（如 `SUPPORTED_METRICS`, `REFERENCE_METRICS`），不仅存在双重维护风险，也无法在单图评估场景中调用新上线的 `directional_clip` 指标。

M8-02 的核心目标：
1. **单一真实源对接**：重构 `image_evaluator/core.py`，所有能力集合（`SUPPORTED_METRICS`, `PROMPT_METRICS`, `REFERENCE_METRICS`, `DATASET_METRICS`, `PAIRWISE_METRICS`, `SOURCE_PROMPT_METRICS`）统一通过 `image_evaluator.registry.list_metrics()` 动态派生；
2. **入参签名扩展**：在 `evaluate()` 签名中新增 `source_prompt: str | None = None`，无缝支持文本引导编辑与风格迁移评估；
3. **严格防御与校验**：
   - 当请求 `directional_clip` 时，必须提供 `reference`, `prompt`, `source_prompt`，任一缺失即抛出明确的 `ValueError`；
   - 拒绝向 `directional_clip` 传入目录路径，明确提示其为单图评估指标；
4. **延迟加载与按需实例化**：仅在实际计算 `directional_clip` 时才延迟导入并实例化 `DirectionalClipPredictor`，确保不引入额外的顶层导入开销；
5. **完全向后兼容**：现有所有指标调用方式、单指标浮点数返回或字典返回结构完全保持不变。

---

## 3. 核心改动明细

### 3.1 `image_evaluator/core.py`
- 导入 `from image_evaluator.registry import list_metrics`；
- 动态派生指标能力集合，替代原静态硬编码集合：
  ```python
  _SPECS = list_metrics()
  SUPPORTED_METRICS = {s.id for s in _SPECS}
  PROMPT_METRICS = {s.id for s in _SPECS if "prompt" in s.inputs.required}
  REFERENCE_METRICS = {
      s.id for s in _SPECS
      if "reference_image" in s.inputs.required
      or "reference_collection" in s.inputs.required
  }
  SOURCE_PROMPT_METRICS = {
      s.id for s in _SPECS if "source_prompt" in s.inputs.required
  }
  PAIRWISE_METRICS = {
      s.id for s in _SPECS
      if "reference_image" in s.inputs.required
      and "arithmetic_mean_for_directory_inputs" in s.aggregation
  }
  DATASET_METRICS = {
      s.id for s in _SPECS if "reference_collection" in s.inputs.required
  }
  ```
- 扩展函数签名：
  ```python
  def evaluate(
      metrics: str | Sequence[str],
      image: Any,
      reference: Any = None,
      prompt: str | None = None,
      source_prompt: str | None = None,
      device: str | torch.device | None = None,
      detailed: bool = False,
      **kwargs: Any,
  ) -> dict[str, Any] | EvaluationResult:
  ```
- 增加针对 `source_prompt` 的校验逻辑，以及单图与目录输入的边界防御；
- 增加对 `directional_clip` 的延迟派发：
  ```python
  if "directional_clip" in selected:
      from image_evaluator.directional_clip_predictor import (
          DirectionalClipPredictor,
      )

      clip_model = kwargs.get("clip_model", "openai/clip-vit-base-patch32")
      pred_dir_clip = DirectionalClipPredictor(
          clip_model=clip_model, device=device
      )
      results["directional_clip"] = pred_dir_clip.evaluate_directional_clip(
          image_src=reference,
          image_edit=image,
          prompt_src=source_prompt,
          prompt_target=prompt,
      )
  ```

### 3.2 `tests/test_api_evaluate.py`
- 新增 6 组完备的单元与集成测试用例，覆盖：
  1. `test_evaluate_directional_clip_in_memory_pil`：PIL 图像单指标与字典返回评估；
  2. `test_evaluate_directional_clip_in_memory_tensor`：PyTorch Tensor 输入评估；
  3. `test_evaluate_directional_clip_validation_errors`：缺失 reference / prompt / source_prompt 异常拦截验证；
  4. `test_evaluate_directional_clip_rejects_directory`：传入目录路径时异常拦截验证；
  5. `test_evaluate_mixed_metrics_with_directional_clip`：与传统配对指标（如 SSIM）混合评估验证；
  6. `test_registry_capabilities_single_source_of_truth`：断言 `core.py` 动态集合与注册中心元数据严格 100% 吻合。

---

## 4. 验证证据链与原始日志

所有原始日志与真实截图已持久化存储于 `artifacts/m8-02-evaluate-registry/`：

| 交付物 | 类型 | 说明 / 结果 |
| :--- | :--- | :--- |
| `targeted_tests.log` | 原始日志 | `tests/test_api_evaluate.py` **27 项测试全部通过** (耗时 16.28s) |
| `full_pytest.log` | 原始日志 | 全量测试套件 **314 项测试全部通过**，0 失败 |
| `ruff.log` | 静态检查 | Ruff 代码规范与风格检查 **All checks passed!** |
| `git_diff_check.log` | 格式合规 | Git diff 空白符与格式检查 clean (0 警告) |
| `git_status.log` | 范围受控 | 仅修改白名单内文件，无意外污染 |
| `heavy_import_isolation.log` | 架构隔离 | 顶层导入无 torch/cleanfid/clip，按需实例化验证无误 |
| `terminal_validation.png` | 真实物理截图 | macOS Terminal.app 执行真实命令截图 (`screencapture -l`) |
| `browser_desktop.png` | 真实文档截图 | Chrome Headless 桌面视口 (1280x1100) 渲染看板截图 |
| `browser_narrow.png` | 真实文档截图 | Chrome Headless 移动窄屏视口 (375x1400) 渲染看板截图 |
| `visual_acceptance.html` | 可视化看板 | 自包含深色模式交互式验收页面 |

---

## 5. 独立验收审查意见 (Independent Acceptance Review)

- **审查员**: Independent Acceptance Reviewer (Antigravity QA / Audit Mode)
- **审查结论**: **ACCEPTANCE READY (建议 Boss 批准验收)**
- **细项评审记录**:
  1. **边界契约吻合度**: `core.py` 完全移除了重复的静态指标名单，由 `registry.py` 统一分发，完全消除了双头维护风险；
  2. **API 向后兼容**: 原有 `evaluate(metrics, image, ...)` 调用顺序与默认字典返回保持不变，老测试用例无一修改且全数通过；
  3. **参数防御充分性**: 对 `source_prompt` 的空值、缺失以及目录入参均具备前置异常拦截，错误提示清晰明确；
  4. **性能与隔离性**: 导入顶层 `image_evaluator` 不触发重量级模型加载；仅在使用 `directional_clip` 时才延迟加载 CLIP 权重；
  5. **截图合规性**: 拒绝任何 PIL/Canvas/AI 伪造截图，Terminal 截图来自真实系统 WindowID 物理抓取，Chrome 截图来自 Headless 真实 DOM 渲染，已完成桌面与移动端响应式检查。

---

## 6. 治理与停机指令
根据项目治理总则与交接文件约定：
- **技术 PASS 不等于 Boss 验收**；
- 在 Boss 明确确认验收 M8-02 之前：
  - **严禁**进入 M8-03；
  - **严禁**执行 Git Commit 或创建分支；
  - **严禁**修改 `pyproject.toml` 版本号（维持 `0.4.0`）；
  - **严禁**发布至 PyPI。
