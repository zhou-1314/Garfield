# P0/P2 诊断 · 复现说明（2026-09-08）

**性质**：低成本 CPU 研究脚本。**不是端到端模型实现**，不碰生产训练链、不跑 P3、不用 GPU、不下载数据。
所有写入都在本目录内；**9/7 的数据与结果只读、未修改**。

## 1. 数据路径（只读）
- 计数与元数据：`../2026-09-07-composition-state-pilot-data/`（`pilot_counts.npz`、`pilot_cells.csv`）
- 9/7 对拍基准：`../2026-09-07-composition-state-pilot-analysis/`（`results_compstate.csv`、`run_compstate_pilot.py`）
- **来源与许可**：GSE164378（Hao et al., *Cell* 2021）。**再分发许可未核，目录内无 licence 文件 ⇒ 记为 UNKNOWN**，未断言可再分发。

## 2. Python 依赖（均为环境既有，**未安装任何新包**）
`/opt/conda/bin/python` 3.10 · numpy 1.26.4 · pandas 2.3.3 · scipy 1.15.3。
`pandas.to_markdown` 需要的 `tabulate` **不可用**，故表格一律用 `to_string` 输出，**未因此安装依赖**。
线程上限固定为 4：`OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS=4`（脚本内设置并写入 manifest）。

## 3. 复现三步
```bash
cd docs/research/2026-09-08-p0-p2-diagnostics
/opt/conda/bin/python p0_admission.py                  # 1) P0 准入 -> p0_admission.json
/opt/conda/bin/python run_p2_diagnostics.py            # 2) P2 诊断 -> results_p2.csv 等（加 smoke 参数可跑小样例）
/opt/conda/bin/python summarize_p2.py                  # 3) 汇总   -> RESULTS_P2.md + summary_*.csv
```
可选：`/opt/conda/bin/python verify_alignment.py` —— 与 9/7 nominal 臂做**实测对拍**（非 by-construction 断言）。
实测耗时：compute 16.7s / load+setup 13.6s / total in-process 30.3s（shell 墙钟另计），峰值 RSS 约 3.36 GB。

## 4. 诊断臂的权限说明（**解释边界，不可省略**）
| 臂 | 额外获得的信息 | 可否当部署成绩 |
|---|---|---|
| `D1_truepi_rna` / `D1_truepi_joint` | **真 π（仅比例，不含 N）** | **否** —— oracle 诊断，**不是性能上界** |
| `D2_cross_full` | 训练侧全参考池（基线） | 是（部署条件） |
| `D2_same_donor` | **目标供体自身参考池** | **否** —— 特许输入 |
| `D2_cross_matched` | 训练侧下采样至同规模 | 是（作为 same_donor 的规模匹配对照） |
| `D3_stage1_joint` | 无 | 是 |
| `D3_stage2_rna` | 无（只用 stage1 的 π̂） | 是 |

- **固定 π 的约束是齐次的**：只锁粗层比例，**不锁尺度**；总量由数据拟合，**真实细胞数 N 从未进入任何求解器**。
- **D2 的 (b)/(c) 差异只说明"参考来源条件改变"**，供体身份与批次/捕获/组成在本设置**未分离**，不得单凭结果归因生物学供体身份。
- **口径**：主报告统一**场景等权**；与 P1 并表时两侧选同一口径即可（P1 的 `tables_p1_calibers_comparison.csv` 已含 cal1 场景等权 / cal2 逐行等权）。
- **单位**：`summary_s_support.csv` 的 `rows`/`s_defined`=**5,280** 是**粗类型评估条数**（1,320 spots × 4 粗类型）；**spot 数是 1,320**。
- 仅 nominal 臂（depth=1.0, dropout=0.0）；2 供体、seed 与 pseudo-spot 为**技术重复**，仅描述，无 margin、无显著性检验。

## 5. 修订史
见 `REVISIONS.md`：rev0 协议冻结版 / rev1 事后工程修订（约束超限门、Frobenius 明示、运行时分列）/ rev2 汇总缺陷修复（`SPOTKEY` 漏 `delta,eta`）。**不冒称一次做对。**

## 6. 哈希
代码与输出的真实哈希见 `manifest_p2.json` 的 `code_sha256` / `output_sha256` / `input_sha256` / `old_artefact_sha256`
（`frozen_at` 为冻结时刻）。**哈希只记录当时状态，不代表与任何历史 gold 比对通过。**

## 7. 独立复核（非本人）
- 复核人：spamot-7（非作者）。**结论：运行层签核通过；B-3（θ 8 维口径）已关闭，顶部条目全部 accept。**
- 报告：`/root/.ao/data/worktrees/spamot/spamot-7/docs/research/2026-09-08-p0-p2-execution-review.md`
  SHA256 `03f7d6ffa76e2ed2eb44f5ceaece97956452de262dc9e2bcd77f81adf193f896`
- 复核已独立验证：齐次约束只锁比例不锁尺度（y→αy 时 π̂ 变化 0/0/2.8e-17、T/T₀ 精确 = α）；
  `solve_fixed_pi` 只接 `pi_given`，**`pi_true` 不出现在 D3 拟合路径**；θ 8 维重构；分母与行数逐一相符。
- **复核通过 ≠ 算法有效、≠ 有增量、≠ 可发表。** 本目录产出为**探索性诊断**，解释边界见 §4。

## 8. Final 稿必须保留的两条限制
1. **S1 只是名义上的"仅组成变化"**：`delta` 取极端值时某粗类型只分到 5 细胞，取整使该类型**真实 s = 2/5 = 0.4** 而非 0.5
   ⇒ **不能称严格独立因素设计**。按**实际计数**评分仍然合法（本轮真值本就由实际计数重算），故**不重跑**。
2. **关于 M2 的先验优势**：只能表述为"在名义设计与实际计数下，**多数**粗类型的真实 s 不变或接近 0.5"，
   **不得**写成"另外三类在所有场景恒为 0.5"。
