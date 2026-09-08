# P0–P2 整合索引

本目录汇总 P0 准入、P1 基线审计与 P2 诊断结果；主结论见 [RESULTS.md](RESULTS.md)。P2 作者原始运行说明保留为 [README_P2.md](README_P2.md)。

## 复现
权威运行位置为 spamot-2 作者目录：
`/root/.ao/data/worktrees/spamot/spamot-2/docs/research/2026-09-08-p0-p2-diagnostics/`
使用 `/opt/conda/bin/python`（numpy 1.26.4、pandas 2.3.3、scipy 1.15.3），在该目录按 README_P2 中三步命令执行；本整合目录不改脚本逻辑。

## 产物
- [P0准入](p0_admission.json)
- [P1场景等权表](tables_p1_nominal_test.csv)
- [P1双口径表](tables_p1_calibers_comparison.csv)
- [P2场景等权主表](summary_macro_scenario_equal_weight.csv)
- [P2按类型表](summary_by_type.csv)
- [P2按供体表](summary_per_donor.csv)
- P2结果大表不随PR分发（按复现步骤生成）
- [P2失败表](failures_p2.csv)
- [P2清单](manifest_p2.json)
- [独立复核](2026-09-08-p0-p2-execution-review.md)

P2 结果为技术伪点评估，非独立生物重复；生产源码未修改，P3未启动。
