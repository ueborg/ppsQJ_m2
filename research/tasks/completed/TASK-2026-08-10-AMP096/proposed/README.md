# proposed/ — state changes awaiting the human gate

**Nothing here is applied and nothing here is authoritative.** These are
candidate corrections to `research/state/**` for the researcher to merge, or
reject, at the single-writer merge step.

All four are **metadata corrections**. None changes a scientific value, an
epistemic status, an amplitude, an exponent or a dispute position. They are
listed here as diffs-in-prose rather than as YAML, because writing them as YAML
files invites someone to copy them into state without review.

---

## P1 — `CB-AMP-096-001` (claim, `withdrawn`, stays withdrawn)

**`evidence_note` is false.** It reads "Original supporting analysis not located
on disk. Historical record only." The analysis **is** on disk and tracked at
commit `6c9c843`: `analysis/lambda_c_phi_analysis.md`, mtime 2026-05-19,
carrying the method, the extrapolation table and the fitted values.

**`confidence_basis` is a single cause where the record shows two steps.** It
reads "was an r_c-type prefactor mislabelled as lambda_c". Proposed replacement,
two-part: (a) generated May 2026 as an inverse-variance-weighted log-log fit to
1/sqrt(L)-extrapolated **lambda_c** values over zeta ≤ 0.3; (b) reinterpreted as
an r_c prefactor on 2026-06-10 per `theory/Y_ZETA_DERIVATION.md:191`
(provenance, not support).

Both halves are attested. The current note asserts only (b) and reads as though
(a) never happened.

## P2 — `OBS-BLKMR-001` (observable)

**`recomputable_from` is false.** It asserts `CMI_mean` and `S_AB_mean` are
"both stored in every guided aggregate". `S_AB_mean` is **absent** from
`DATA_INTERNSHIP/Data/pps_aggregates/agg_caseB_combined.pkl` — the dataset
`EV-DATA-AGGCASEB-001` and `CB-AMP-001` rest on. Present keys: `B_L_err`,
`B_L_mean`, `CMI_err`, `CMI_mean`, `ESS_mean`, `L`, `N_c`, `S_err`, `S_mean`, …

**`open_task` is partially discharged**, on `agg_pps_clone_guided_prod.pkl`
(877 records) **only**: the two locators agree to a median 0.31% on crossing
position and ~1% on amplitude. Scope the record explicitly; it was **not**
performed on `EV-DATA-AGGCASEB-001`, which cannot support it.

## P3 — `SRC-KMR-2023` (source) — the one that matters

**Withdraw the `invoked_for` statement** "the reported diffusive-case boundary
lambda_c ~ A*r_c^{1/2} with A of order one half". Full-text inspection of
SciPost Phys. 14, 031 (all 1758 extracted lines) finds:

- no statement of that form anywhere;
- no occurrence of the string "diffusiv" in body, captions or references;
- KMR's `r_c` is the **detector readout threshold** of Eq. (20), taking values
  −2, −1.5, −1, −0.5 in Fig. 5 — not lambda_c/(1−lambda_c);
- no numerical value at all for the gamma = 0 critical alpha/w.

**Upgrade** the B_L definition from title-only to body-verified: the main text
(Sec. 3.2) reads "we use combination B_L = S̄_L S̄_L^{top}", confirming
`OBS-BLKMR-001` is the product of trajectory-**averaged** quantities.

This is a **symbol collision across the internal/external boundary**, and it is
the origin of the M1 manuscript's "twice the diffusive prefactor" sentence. The
comparison is not wrong; it is **not well-posed**. The manuscript is out of scope
here and was not modified.

## P4 — `SRC-FULGA-2012` (source)

Identified: arXiv:1205.1441, Fulga, Akhmerov, Tworzydło, Béri, Beenakker,
"Thermal metal-insulator transition in a helical topological superconductor",
PRB 86, 054505 (2012). The state file records `local_copy: false`, arXiv id
UNKNOWN and a title "REMOVED because it was a guess". Retrieved and read.

Table I gives the class-DIII thermal metal-insulator nu for a 2D disordered
Chalker–Coddington network. **No measurement, monitoring, replica or Born rule
appears in the paper.** `DEC-CITATION-001` item 3 is now verified from both
sides, with the nuance that Jian's Born and forced nu values are statistically
compatible and their class separation rests on x(1), x(2) and zeta_1 rather than
on nu.

---

## Not proposed, deliberately

- No change to any `epistemic_status`, `confidence`, amplitude or exponent.
- No change to `DISP-PHI-001` or `DISP-WINDOW-001`. Both stay open.
- No new claim. The surviving C4 result is a scoped note on an existing
  observable, not a claim.
