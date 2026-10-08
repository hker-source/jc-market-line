# HKJC Odds Research — Methodology Record

**Status: PRE-REGISTRATION v2 — REGISTERED 2026-10-08.**
v2 supersedes v1 by registering the deployment cadence and the velocity
threshold↔cadence correspondence. Vocabulary (§1), schema (§2), model (§3),
sensitivity (§4), sample gates (§5), and confirmed decisions (§6) from v1 remain
frozen; no v1 coefficient or rule is retroactively altered, only the cadence
context is registered.
Any change to vocabulary, features, or decision rules requires a new version (v3).

---

## 1. Controlled vocabulary (terminology lock)

Purpose: eliminate the collision where "steam" means two different things
(velocity-based event label vs. λ-sign descriptive bucket).

| Concept | Reserved term | Definition / source |
|---|---|---|
| Sharp / fast move (EVENT) | `steam` | `odds_events.movement_type='steam'` (velocity ≥ threshold), pre-KO |
| Slow move (EVENT) | `drift` | `odds_events.movement_type='drift'` |
| Small move (EVENT) | `micro` | `odds_events.movement_type='micro'` |
| Price direction (DESCRIPTIVE) | `shortened` / `flat` / `lengthened` | sign of λ vs ±τ |
| Probability drift | `delta_prob` (Δ) | `p1 − p0` |
| Relative probability drift | `kappa` (κ) | `Δ / p0` (winsorized, secondary) |
| Log-odds price drift | `lambda` (λ) | `ln(o0) − ln(o1)`, nats |

Rules:
- **R1** — "steam" means ONLY the velocity-based event label. Never call a λ>τ
  group "steam"; call it the **shortened** bucket.
- **R2** — descriptive λ groups: `shortened` (λ>τ), `flat` (|λ|≤τ),
  `lengthened` (λ<−τ).
- **R3** — "drift" (noun) is the slow-move EVENT label only. For price
  direction write "shortened"/"lengthened".
- **R4** — every report names the facet explicitly: event-label
  (`steam`/`drift`/`micro`) vs price-direction (`shortened`/`flat`/`lengthened`).
- **R5** — τ is pre-registered at **0.02 nats**, frozen. Report τ∈{0.01, 0.05}
  as sensitivity; the headline uses 0.02.

---

## 2. Schema verification (verified 2026-10-07 against `hkjc_odds.db`)

Resolves the draft-vs-raw naming discrepancy:

- Raw JSON selection key `selId` → stored column **`sel_id`**.
- Raw JSON line key `lineId` is **absent** in the standalone `matchResultDetails`
  payload → stored **`line_id` = NULL** there; present only when the live
  `resultOnly` query selects it.
- `line_index` = 0-based position of the line within its pool; **always populated**.
- Unique index on `match_settlements`:
  `(match_id, odds_type, COALESCE(pool_id,''), line_index, COALESCE(comb_str,''), COALESCE(sel_id,''))`
  → uses `line_index`, **not** `line_id`, because SQLite treats NULLs as distinct
  and `line_id` is frequently NULL.
- ⚠️ **`line_index` is NOT stable across captures.** It is the within-capture
  position of a line in the payload, and HKJC may reorder lines between polls.
  Use it ONLY for within-capture de-duplication. **NEVER use `line_index` as a
  cross-capture join key** — join on `line_id` when the source query provides it,
  otherwise on `comb_str` with the single-line assumption.
- Odds side key: `(match_id, odds_type, pool_id, line_id, comb_id)`; `comb_str`
  lives in `odds_raw`.
- **JOIN rule**: odds ↔ settlement on
  `(match_id, odds_type, pool_id, normalize(comb_str))` + `line_id` when the
  settlement carries it; otherwise accept only when that `(pool, comb)` is
  single-line.
- `comb_id` is numeric (`'1','2','3'`); `comb_str` is the semantic key
  (`'H','A','D','01','H:H'`). `normalize()` strips leading zeros on numeric strings.
- Verified columns:
  - `match_settlements` = id, match_id, match_internal_id, odds_type, pool_id,
    inst_no, pool_status, line_index, line_id, line_condition, comb_str,
    comb_status, win_ord, sel_id, sel_str, sel_name_en, sel_name_ch, captured_at.
  - `match_results` = id, match_id, match_internal_id, stage_id, result_type,
    home_result, away_result, result_confirm_type, payout_confirmed, captured_at.
- Reality: settlement currently holds **3 rows (FB5809 HAD only)** → n=3.

---

## 3. Frozen model & features

```
M0:     logit P(WIN) = β0 + β_λ·λ + β_x·ln(o0) + Σ_m γ_m·1[market=m] + β_s·steam_dummy
        ; cluster-robust SE by match_id (CR2 + wild-cluster bootstrap when G<50)
Mfull:  M0 + β_int·(λ·ln(o0)) + β_sλ·(steam_dummy·λ)
```
- `Y` = WIN(1) / LOSE(0); VOID/push excluded (counted separately).
- `π = (o0−1)·WIN − LOSE`; `EV = P̂·o0 − 1`.
- Primary regressor **λ** (raw odds); κ winsorized P1/P99 per `odds_type` (secondary).
- SGA: kept, raw `1/odds`, `is_sga` flag + `odds_type` FE (never normalized).
- Bet price = **opening `o0`**; stake = **flat** (Kelly deferred).

---

## 4. Pre-registered sensitivity rules

Policy: **M0 is CONFIRMATORY. S1–S10 are EXPLORATORY**, Holm-corrected within the
family. A sensitivity model may OVERRIDE M0 **only** if its pre-registered
falsification fires. Always report both; no post-hoc model shopping.

| ID | Model | Pre-registered falsification → action |
|---|---|---|
| S1 | λ → κ (winsorized) | sign(β) flips vs M0 → claim = "regressor-choice sensitive / inconclusive"; do NOT adopt the favourable one |
| S1b | λ → Δ | same as S1 |
| S2 | + λ·ln(o0) | interaction CI excludes 0 → headline becomes the conditional drift effect at median ln(o0); report the curve, not one β_λ |
| S3 | + steam_dummy·λ | interaction CI excludes 0 → report separate steam-λ and drift-λ slopes; else keep single λ |
| S4 | path decomposition λ_steam+λ_drift+λ_inplay | descriptive only; cannot override M0 |
| S5 | + peak\|velocity\|, mean accel | descriptive; non-causal; cannot override M0 |
| S6 | exclude SGA | β_λ sign flips OR \|Δβ_λ\|/\|β_λ\| > 25% → split SGA into its own model; report both; pooled result demoted |
| S7 | two-way cluster (match, match×market) | SE ratio (two-way/one-way) > 1.5 → report the WIDEST CI; point estimate unchanged |
| S8 | ln(o0)-decile within-stratum effect | sign disagrees with pooled β_λ → functional-form concern; headline = "inconclusive" |
| S9 | per-market M0 | ≥2 markets with opposite-sign β_λ each CI excluding 0 → report heterogeneity; do NOT pool |
| S10 | control swap ln(o0) → p0 (devig) | β_λ changes materially → keep ln(o0) (primary); report swap as sensitivity |

Multiple testing: only M0's β_λ is confirmatory; all others exploratory with Holm
across {S1, S1b, S2, S3, S6, S8, S9, S10}.

---

## 5. Sample gates

- **EPV ≥ 10** → ≥ 90 WIN and ≥ 90 LOSE for M0.
- **Gate B** → ≥ 150 WIN / ≥ 150 LOSE, ≥ 100 matches, ≥ 8/8 per market,
  ≈ ≥ 1,000 settled selections.
- **Cluster gate** G ≥ 50 settled matches (target 100).
- **Power**: OR 2.0 ≈ 3,000 selections; OR 1.5 ≈ 9,000.
- **Interim**: while n small, print "DATA INSUFFICIENT"; all coefficients = NOISE.

---

## 6. Decisions — CONFIRMED (owner sign-off 2026-10-07)

- **λ primary** — CONFIRMED. SGA-safe and longshot-stable; raw-odds based, so it
  is independent of the de-vigging method.
- **SGA keep-with-FE + mandatory S6** — CONFIRMED. Maximizes sample size; S6
  (exclude-SGA) is the mandatory robustness safeguard (see §4).
- **Bet price = opening `o0`** — CONFIRMED. The objective is to test whether the
  λ signal has predictive power **relative to the opening price**, not to test
  execution timing; this matches the current project phase.
- **Flat stake, Kelly deferred** — CONFIRMED. Keep staking simple until an edge is
  confirmed; Kelly is deferred to a later phase.

Phase 1 (preprocessing module + DQ checks) is authorized by this sign-off.

---

## 7. Deployment cadence & velocity threshold correspondence (v2, registered 2026-10-08)

### 7.1 Cadence

| Component | Schedule | Mechanism |
|---|---|---|
| Odds poll | every 15 min nominal (cron `*/15 * * * *`) | GitHub Actions `poll.yml`, single poll per run |
| Backup | daily 16:30 UTC | `backup.yml` → Artifact (30-day retention) |
| Watchdog (freshness) | every 30 min nominal | `watchdog.yml`, fail if `scraped_at` > 45 min stale |
| Settlement | daily 07:10 UTC | `settle.yml`, fetch settled matches |

- GitHub cron has **jitter** (nominal ± minutes). Velocity is therefore always
  computed from `seconds_since_last` (actual elapsed between two consecutive
  `scraped_at` values for the same `(match_id, odds_type, pool_id, line_id, comb_id)`),
  **never** from the nominal 15-min interval. This keeps classification unbiased
  under jitter.
- `concurrency: group hkjc-poll` + `cancel-in-progress: true` prevents overlap;
  worst case is loss of one poll (max 15 min gap). `odds_raw` writes are a single
  transaction (no partial rows).
- Watchdog 45-min threshold = 3 consecutive missed 15-min polls — the alarm.

### 7.2 Velocity threshold correspondence

`STEAM_VELOCITY_THRESHOLD = 0.005` (0.5 pp implied-prob per minute) is **frozen
from v1**. At 15-min nominal cadence:

| Actual poll interval | Velocity needed to classify as `steam` |
|---|---|
| 15 min (nominal) | Δprob ≥ 0.005 × 15 = **7.5 pp** per window |
| 10 min (fast GitHub fire) | ≥ 5.0 pp |
| 20 min (slow GitHub fire) | ≥ 10.0 pp |

Because velocity = Δprob / (seconds/60), the threshold auto-scales to the actual
interval. A 7.5 pp move over 15 min classifies as `steam`; the same 7.5 pp over
30 min (0.25 pp/min) does **not** and falls to `drift` (Δprob ≥ 0.02) or `micro`.

### 7.3 Registered thresholds (unchanged from v1)

| Parameter | Value | Source (file:line) |
|---|---|---|
| `STEAM_VELOCITY_THRESHOLD` | 0.005 pp/min | HKJC_odds_modelling15.py:31 |
| `DRIFT_PROB_THRESHOLD` | 0.02 (2 pp cumulative) | HKJC_odds_modelling15.py:32 |
| `MIN_PROB_CHANGE_TO_LOG` | 0.0005 (0.05 pp noise floor) | HKJC_odds_modelling15.py:33 |
| λ τ (price-direction bucket) | 0.02 nats | §1 R5 |

No threshold is recalibrated in v2; this section merely registers how the frozen
v1 values behave under the 15-min deployment cadence.
