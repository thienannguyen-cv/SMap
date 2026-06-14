# Contributing to SMap

Welcome to the SMap Open Source Project! SMap operates on a strict **Source-Completion via App-Based Auditing** paradigm. We don't just write code; we prove it correct mathematically and verify it visually.

## The Two-Phase Model & Agent Inference
SMap alternates between two distinct phases. You do **not** track the phase by hand: it is carried in the project handoff (`harness/context/effective-verbal-context.md`) and surfaced **automatically** the moment an AI agent loads context via the `read-effective-verbal-context` skill — the agent reads the current phase and warns you when an action (e.g. editing source) is out of bounds. If you are only reporting a bug, the phase does not concern you at all. The two phases are:

1. **Initialization Phase:** The source code is considered the "fixed truth" (even if buggy). The goal is to build the `math_model.md` and JSX Simulator around it to understand its exact behavior. In this phase, **the source code is frozen**.
2. **Development Phase:** Once the `math_model.md` adequately covers a scope, the model becomes the source of truth. We use the JSX Simulator to visually expose bugs, and then we **fix the source code** to match the math model. 

## PR / Contribution Types

We recognize 7 types of Pull Requests. 

**Roles (referenced below).** *General Contributor* — anyone; can report bugs, fix code, and improve the JSX UI. *Domain Expert* — a contributor trusted on the math/algorithm; can edit the math model and propose new applications. *Admin / Core Maintainer* — owns merges and the `/harness/` machinery (see [`.github/CODEOWNERS`](.github/CODEOWNERS)). *Unprimed General Contributor* — a General Contributor (human or agent) who runs an audit **without** being shown the bug list or expected answers, so their verification stays independent.

**★ Core Principle (Evidence Beats Rank):** A reproducible falsification (e.g., a working probe script) submitted by *any* contributor overrides seniority. Ranks determine who merges, but evidence determines what is true.

### 1. Report a Bug (Bắt BUG)
- **Role:** General Contributor
- **Action:** Add a new bug to [`active_bugs.md`](active_bugs.md) and ideally provide a probe script (`t_*.py`) that demonstrates the bug.
- **Requirement:** You do not need Admin rights to report a bug. Just provide reproducible evidence.
- **This includes the harness itself:** if the README, this guide, or the simulator left you unsure what to do next, that friction is a legitimate bug — report it so we can make the harness smoother.

### 2. Fix a Code Bug (Fix BUG - Code)
- **Role:** General Contributor / Domain Expert
- **Action:** Fix the source code (`smap/` or `rectify.py`) to perfectly match the current `math_model.md`.
- **Requirement:** Proceed only if the relevant scope is in the **Development Phase**. If the scope is still in the Initialization Phase, source modifications must be deferred to preserve the reference point. You must demonstrate that your code fix resolves the bug.
- **Testing:** Verify convergence locally by running `node harness/simulator/_repro_check.mjs`. The PR must also update the toggle in the JSX Simulator.

### 3. Fix the Math Model (Fix BUG - Math)
- **Role:** Domain Expert / Admin
- **Action:** Modify `harness/specs/math_model.md` when the current mathematical model is fundamentally flawed or incomplete.
- **Requirement:** High scrutiny applied. Your PR description must include the logical derivation or mathematical proof of why the previous model failed and how the new model guarantees convergence. Use formal mathematical notation (LaTeX block) in the PR if necessary.

### 4. Enhance the JSX Simulator (Fix ứng dụng JSX)
- **Role:** General Contributor
- **Action:** Improve the UI/UX of `harness/simulator/smap_simulator.jsx` (the simulator's source; launch it via the dev server in `harness/preview/` — see the README's *Running the JSX Simulator*).
- **Requirement:** These are non-functional requirements (NFRs). Enhancements are driven by user votes and project direction. Adding new visual diagnostic tools is highly encouraged.

### 5. Improve the Harness (Cải thiện Harness)
- **Role:** Admin / Core Maintainer (Domain Experts can propose changes but require Admin review)
- **Action:** Modify the audit scripts (`harness/audit/`), AI agent prompts (`harness/agents/`), or Agent Skills (`harness/skills/`).
- **Requirement:** The verification harness is the core engine of our trustworthiness. Changes here require Admin-level approval and strict testing to ensure we don't break the auditing machinery.
- **Testing the Harness:** Before submitting, you MUST verify the structural integrity of the harness by running `python harness/audit/setup_sandbox.py` to ensure the ephemeral sandbox builds correctly, and run the `t_*.py` probes inside `harness/audit/sandbox/` to ensure no false positives occur.

### 6. Run an Audit Round (Thực hiện Kiểm chứng Độc lập)
- **Role:** Unprimed General Contributor
- **Action:** Write independent probes to test if the math model accurately reflects the source code, serving as a layer of independent verification.

### 7. Propose a New Application/Schema (Đề xuất Ứng dụng mới)
- **Role:** Domain Expert
- **Action:** Submit an RFC to expand the scope of correctness (e.g., adding a new `applications/` scenario) to cycle back into the Initialization Phase.

## How to Submit

> **Which submission path applies to me?**
> - **Everyone on GitHub (humans and agents working against the live repo):** open a normal GitHub **Issue** (bug) or **Pull Request** (fix) and fill in the matching template — [`bug_report.md`](.github/ISSUE_TEMPLATE/bug_report.md) or [`PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md). This is the standard path.
> - **AI agents working offline (no GitHub access yet):** use **Local PR Simulation** below to stage an equivalent PR on disk.

### Local PR Simulation (offline / Admin pre-integration audit)

Local PR Simulation reproduces the GitHub PR experience on disk, so an agent can contribute without a network and so Admins can audit that a process/harness change stays in sync with the standards **before** it lands on GitHub — with no drift in the contributor experience. (This audit has already passed for the current harness integration.) To stage one:

1. **Create a PR Directory:** Create a new directory at the root of the project: `PRs/<PR_ID>/` (e.g., `PRs/fix-pola-bug/`).
2. **Apply the GitHub Template:** Copy the exact contents of [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) into a new file located at `PRs/<PR_ID>/pr_description.md`.
3. **Provide Evidence:** Fill out the `pr_description.md` file completely. You MUST:
   - Provide a `snapshot_*.json` file inside the PR folder if applicable.
   - Categorize the bug (App Inconsistency, Source Bug, or By-Design).
   - Perform the POLA analysis. Ensure you are reasoning strictly from the codebase and math model, not from your training memory.
4. **Include Patches:** Place any modified scripts or patch files inside the `PRs/<PR_ID>/` directory.

A staged Local PR carries the exact same evidence a GitHub PR does, so an Admin can review it identically and then merge it via native GitHub mechanics.

## Getting Started with AI Agents

If you are an AI agent picking up this project cold, load the project's handoff before doing anything else:

1. **Load context.** Read `harness/context/effective-verbal-context.md` (the canonical handoff) via the `harness/skills/read-effective-verbal-context` skill. It recovers the objective, the current phase (see *The Two-Phase Model* above), active constraints, the artifact map, and open work — no prior chat required.
2. **Respect the Evidence Standard.** Reason strictly from the source (`smap/`, `rectify.py`) and `harness/specs/math_model.md`, not from training memory. Tie every claim to a `file:line` anchor or an executed probe.
3. **Pick a contribution type** (above); the current phase is already in the context you loaded in step 1, so honour it rather than assuming. When a change is hard to reverse or the phase is ambiguous, ask the maintainer (see *Questions?* below).
4. **Use Local PR Simulation** (above): create `PRs/<PR_ID>/`, fill `pr_description.md` from the template, attach a `snapshot_*.json` and your probe output.
5. **Verify before you claim.** Run `node harness/simulator/_repro_check.mjs` (the mirror gate — it exits non-zero on a convergence regression) and, for source/probe work, `python harness/audit/setup_sandbox.py` followed by the relevant `t_*.py` inside the freshly built `harness/audit/sandbox/`. Report results faithfully, including failures.

## Questions?

If you are unsure about anything — which phase a scope is in, whether your finding is a real bug, or how to attach a snapshot — open a [GitHub Issue](https://github.com/thienannguyen-cv/SMap/issues) (use the bug template if it might be a bug) or check the project board on [Trello](https://trello.com/invite/b/66d545d4e065eebded9a9c8f/ATTI56f6dabcfab65e388e9fa66b42e77f6bE3EB9A69/smap-project-management). There is no wrong question — confusion about the process is itself useful signal for improving the harness.
