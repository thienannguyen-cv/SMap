# SMap Active Bug Catalogue

> [!IMPORTANT]
> **Reporting a New Bug:** All new bug reports MUST adhere to the SMap Evidence Standard. Please use the `.github/ISSUE_TEMPLATE/bug_report.md` template to ensure you include a `snapshot_*.json` and a POLA analysis. General Contributors can add bugs to this list without needing Admin rights.



## Bug Catalogue (source-confirmed; math_model §4) — each maps to an app toggle + fix
| Bug | Severity | Source anchor | Effect (corrected) | Fix direction |
|---|---|---|---|---|
| **A — FUT missing convergence filter** | **VERIFIED FIX (pending merge)**. | DEP `y_flow*=(1−case)` `rectify.py:294`; FUT lacked it. | At S1 with y_flow==1, alignment fires ⇒ `g_m≠0` ⇒ S1 loses strict mask fixed point. Verified fix makes FUT absorbing (`g_m=0`) via `t_bug_a_minimal.py`. | Fixed: Added `y_flow*=(1.−((pre_mask>OFF)==target))` after FUT L418 |
| **B — k_const ramp discarded** | precision-only at proof (reopen caveat) | `smap.py` k_const=1.0 (L11); ipynb L435 pins `=1.` discarding L432 ramp | higher k = more participation; source pins MAX 1.0 ⇒ no in-place-warming starvation (occlusion converges at k=1.0). **BUT far/k_const stall: coverage-by-routing fails at k=1.0** (forward-observable; schedule the sole lever). m-fixed regime re-derivation pending. | ipynb L435 → `=k_const` AND reconcile L62/L63 |
| **C — app B not source-faithful** | AUDITABILITY (app-side done) | `rectify.py` DEP B is dim=1 cross-channel (L203) vs uniform 3×3 | toggle "B = source-faithful" applies DEP→cross-channel→self at finest | app-side complete |

| **D — z-buffer mask-fallback is a value no-op** | CORRECTNESS (Source Bug) | `smap.py` L188 clobbers `weights4` | L189 fallback uses winner's mask instead of center's. A nearer inactive winner uncovers a matched active cell. | Keep center gather in a second variable for L189, or make inactive slots non-competitive |
