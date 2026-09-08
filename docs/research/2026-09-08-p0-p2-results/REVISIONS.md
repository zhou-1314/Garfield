# 工程修订记录（不冒称事前冻结）

## rev0 — 2026-09-08 首次正式运行（协议冻结版）
协议 PROTOCOL_FROZEN.md 在看到任何 P2 结果前冻结；rev0 按该协议跑出 73,920 行、0 失败。
rev0 产物 SHA256（保留，不删除）：
- `run_p2_diagnostics.py` : `9d8a8e555b8158741bac593fd30e53607ecc89b14685c1d7315e385c2e8d073d`
- `results_p2.csv` : `9bf1c34339aeefb82a289de68d7eb6b41eac26e0657911ea7217ffc073494d82`
- `counts_p2.json` : `251e6d211104e9fc57222030a8100aa556831288c268ae54bac44792bdb522be`
- `exact_checks.json` : `3462d8ce7b66263129116ece9d77e3f0ee3872f62140dfcaac348eb65d89151e`

**rev0 全量约束残差实测**：D1 max 1.113e-08、D3 max 4.302e-09，**均已远低于 1e-6，无一超限**。
因此 rev1 的失败处理**不是**为修复已发生的失败而加，**也没有**因此调整 `rho` 或网格。

## rev1 — 2026-09-08 工程修订（**事后添加，非事前冻结**）
按 root 指示补三项，**均不改变数值路径**：
1. **约束超限门**：`solve_fixed_pi` 返回的 `resid > 1e-6` 时，该 spot-臂记为失败，`reason=constraint_violation`，
   计入 failed / 排除分母，**不写入结果表**。（rev0 中该分支从未触发，见上。）
2. **范数明确为 Frobenius**：设计矩阵与 ADT 块的 `np.linalg.norm(2-D array)` 即 Frobenius 范数，文档与 manifest 中写明。
3. **运行时字段分列**：`runtime_compute_sec` / `runtime_load_and_setup_sec` / `runtime_total_in_process_sec` 三者分列，
   **不得混称**；shell 墙钟另计（含解释器启动与 import）。
`rho` 与网格 **未改**。rev0 的 CSV 保留为 `results_p2_rev0.csv`。

## rev2 — 2026-09-08 汇总脚本缺陷修复（**我方自查发现，非审核指出**）
**缺陷**：`summarize_p2.py` 的 `SPOTKEY` 漏了 `delta` 与 `eta`。`spot_id` 只在**单个网格点内**取 0..39，
而 S1/S2/S3 分别有 3/3/4 个网格点 ⇒ 同一 `spot_id` 的不同网格点被 groupby **合并成一个"spot"**。
**可观察症状**：每个场景每供体都恰好 60 个"spot"（S0 真值即 60，S1/S2/S3 被压缩到 60），
且"场景等权"与"逐行等权"两种口径数值**完全相同（差 1e-16）**——这正是被掩盖的证据。
**影响范围**：仅影响 `summarize_p2.py` 的分组与两种口径的可区分性；
`results_p2.csv`（73,920 行原始输出）**不受影响，无需重跑诊断**。
由于 MAE/RMSE 都是对绝对/平方误差取均值，rev1 报出的**数值本身仍是合法的误差均值**，
但它是"跨网格点汇合"后的均值，**不是按协议的场景等权**，且使两种口径无法区分。
**修复**：`SPOTKEY` 补入 `delta`、`eta`；仅重跑汇总。rev1 的表已被覆盖，**此处如实记录，不冒称一次做对**。
