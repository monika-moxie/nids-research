# Decisions

## Decision Log

### 2026-08-20 - Phase 0 scaffold layout

**Decision:** Use a modular top-level structure with separate directories for shared pipeline work, CI3201 adversarial robustness, CI3203 federated learning, the bridge experiment, and the paper.

**Why:** The research project has two course-facing parts plus one integration question. Keeping these boundaries visible makes the implementation easier to defend, test, and assign to presenters.

**Tradeoff:** This creates a few more folders up front, but it prevents later experiments from becoming tangled in one large script directory.
