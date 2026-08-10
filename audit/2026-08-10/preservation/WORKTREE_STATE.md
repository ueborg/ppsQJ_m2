# Working-tree preservation record

Captured 2026-08-10 14:00 +03. READ-ONLY. No git state was altered.

## HEAD and branches
```
9b617fad03ded5737e9f1bf1610ffeb2a50a0996
  claude/nifty-boyd 6e6732c Correct jump behavior
* main              9b617fa [origin/main] Add Ruche submit script for guided Case A grid (dense lambda window around 0.5, L up to 128)

remote: https://github.com/ueborg/ppsQJ_m2.git
upstream ahead/behind:
0	0
```

## git status --short
```
 M pps_qj/parallel/grid_caseA.py
 M pps_qj/parallel/worker_caseA.py
 M pps_qj/parallel/worker_clone_pps.py
 M scripts/habrok/pps_scan/README.md
 M scripts/habrok/pps_scan/submit_clone_guided_nu.sh
 M theory/HANDOFF.md
?? analysis/_reanalyze_parity.py
?? analysis/agg_guided.py
?? analysis/aggregate_guided.py
?? analysis/bdg_actual_vs_simple.png
?? analysis/binder_allzeta_collapse.png
?? analysis/binder_analysis_v2.png
?? analysis/binder_crossings.png
?? analysis/binder_crossings_clean.png
?? analysis/binder_crossings_zoom.png
?? analysis/binder_explained.png
?? analysis/binder_fss_proper.png
?? analysis/binder_honest.png
?? analysis/binder_runAfast.png
?? analysis/binder_zeta1_full.png
?? analysis/caseA_diag.py
?? analysis/cross_real_variance.png
?? analysis/crossings_B.py
?? analysis/crossings_sanity.py
?? analysis/delta_B_diagnostic.py
?? analysis/entropy_scaling_zeta0.py
?? analysis/extrapolation_per_zeta.png
?? analysis/fit_and_caseA.py
?? analysis/fit_caseA.py
?? analysis/fit_nu_zeta1.py
?? analysis/gc_phi_eff.png
?? analysis/gc_scaling_test.png
?? analysis/gc_scaling_test.py
?? analysis/global_fss.json
?? analysis/global_fss.png
?? analysis/global_fss_merged.json
?? analysis/global_fss_merged_v2.json
?? analysis/global_fss_results.json
?? analysis/global_fss_v2.png
?? analysis/lambda_c_phi_analysis.md
?? analysis/lambda_c_phi_fit.png
?? analysis/lc_compare_observables.png
?? analysis/lc_functional_forms.png
?? analysis/liouvillian/
?? analysis/noclick_spectrum_probe.py
?? analysis/nu_assessment.png
?? analysis/other_observables.png
?? analysis/parity_sweep.log
?? analysis/phase2_saturation_check.py
?? analysis/phase_boundary_final.png
?? analysis/phase_boundary_honest.png
?? analysis/phase_boundary_merged.png
?? analysis/phase_diagram_data.json
?? analysis/phase_diagram_final.png
?? analysis/phase_diagram_theory.png
?? analysis/phase_diagram_v1.png
?? analysis/phase_diagram_v2.png
?? analysis/phi_fit_diagnostic.png
?? analysis/plot_binder_allzeta.py
?? analysis/plot_binder_allzeta_v2.py
?? analysis/plot_binder_allzeta_v3.py
?? analysis/plot_binder_crossings.py
?? analysis/plot_binder_explained.py
?? analysis/plot_binder_proper.py
?? analysis/plot_other_observables.py
?? analysis/renyi_washout.py
?? analysis/theta1_scaling.png
?? analysis/tmp_explore.py
?? analysis/two_param_collapse.png
?? analysis/var_reduction/
?? analysis/xi_ps_bulk_dispersion.png
?? analysis/xi_ps_verification.png
?? analysis/zeta0_benchmark_analysis.md
?? audit/
?? logs/
?? pps_qj/backward_pass_sector.py
?? pps_qj/cloning.py.bak_guided
?? pps_qj/cloning.py.bak_prelowrank
?? pps_qj/cloning.py.bak_spawn
?? pps_qj/cloning_caseA.py.bak_guided
?? pps_qj/cloning_caseA.py.bak_spawn
?? pps_qj/gaussian_backend.py.bak_guided
?? pps_qj/gaussian_backend.py.bak_prelowrank
?? pps_qj/gaussian_backend.py.bak_prenewton
?? pps_qj/gaussian_backend_caseA.py.bak_guided
?? pps_qj/parallel/grid_pps.py.bak_guidedgrid
?? pps_qj/parallel/worker_caseA.py.bak_guided
?? pps_qj/parallel/worker_clone_pps.py.bak_guided
?? pps_qj/parallel/worker_clone_pps.py.bak_prelowrank
?? saturation_output/
?? scripts/aggregate.py
?? scripts/make_benchmark_figures.py
?? scripts/run_exact_benchmark.py
?? scripts/run_sweep_l4.py
?? scripts/validate_cloning.py
?? scripts/validate_jump_distribution.py
?? scripts/validation_cloning.pdf
?? scripts/validation_cloning.png
?? scripts/validation_jump_distribution.pdf
?? scripts/validation_jump_distribution.png
?? slurm/submit_nu_zeta1.sh
?? tests/conftest.py
?? tests/test_backward_pass_sector.py
?? tests/test_exact_benchmark.py
?? theory/HANDOFF.md.bak_20260617
?? theory/VARIANCE_REDUCTION.md
?? theory/archive/HANDOFF.md.bak_20260607_ladder
```

## git diff --stat (tracked, unstaged)
```
 pps_qj/parallel/grid_caseA.py                     |  72 ++++++++++++
 pps_qj/parallel/worker_caseA.py                   |  16 +++
 pps_qj/parallel/worker_clone_pps.py               |  52 +++++++++
 scripts/habrok/pps_scan/README.md                 |  31 +++++
 scripts/habrok/pps_scan/submit_clone_guided_nu.sh |  30 ++++-
 theory/HANDOFF.md                                 | 132 ++++++++++++++++++++++
 6 files changed, 328 insertions(+), 5 deletions(-)
```

## Untracked source/theory/test files (would be LOST from a clean clone)
```
analysis/_reanalyze_parity.py
analysis/agg_guided.py
analysis/aggregate_guided.py
analysis/caseA_diag.py
analysis/crossings_B.py
analysis/crossings_sanity.py
analysis/delta_B_diagnostic.py
analysis/entropy_scaling_zeta0.py
analysis/fit_and_caseA.py
analysis/fit_caseA.py
analysis/fit_nu_zeta1.py
analysis/gc_scaling_test.py
analysis/lambda_c_phi_analysis.md
analysis/liouvillian/RESULTS.md
analysis/liouvillian/plot_and_analyze.py
analysis/liouvillian/pps_lindbladian.py
analysis/liouvillian/scan_critical.py
analysis/liouvillian/two_replica.py
analysis/liouvillian/two_replica_scan.py
analysis/noclick_spectrum_probe.py
analysis/phase2_saturation_check.py
analysis/plot_binder_allzeta.py
analysis/plot_binder_allzeta_v2.py
analysis/plot_binder_allzeta_v3.py
analysis/plot_binder_crossings.py
analysis/plot_binder_explained.py
analysis/plot_binder_proper.py
analysis/plot_other_observables.py
analysis/renyi_washout.py
analysis/tmp_explore.py
analysis/var_reduction/activity_cv.py
analysis/var_reduction/bottleneck_test.py
analysis/var_reduction/chunk_bias_cert.py
analysis/var_reduction/controlled_sampler.py
analysis/var_reduction/coupled_snapshot_pilot.py
analysis/var_reduction/coupling_cmi_kmr.py
analysis/var_reduction/coupling_lambda.py
analysis/var_reduction/crossing_prod.py
analysis/var_reduction/d2_scaling.py
analysis/var_reduction/galerkin_control.py
analysis/var_reduction/l64_reference.py
analysis/var_reduction/meanone_gate.py
analysis/var_reduction/nc_ladder.py
analysis/var_reduction/scgf_cv.py
analysis/var_reduction/snapshot_gain.py
analysis/var_reduction/suffix_mcmc_pilot.py
analysis/var_reduction/threearm_benchmark.py
analysis/var_reduction/traj_common.py
analysis/var_reduction/xo_lagged.py
analysis/zeta0_benchmark_analysis.md
audit/2026-08-10/00_STAGE1_FINDINGS.md
audit/2026-08-10/01_CLAIM_LEDGER.md
audit/2026-08-10/02_DATA_INVENTORY.md
audit/2026-08-10/03_AMPLITUDE_TRACE.md
audit/2026-08-10/04_MISSING_RESULTS.md
audit/2026-08-10/preservation/WORKTREE_STATE.md
audit/2026-08-10/recovered_ephemeral/MANIFEST_TABLE.md
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/anchor128.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/ascan.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/blocker.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/controlled_cloning.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/controlled_sampler.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/csampler2.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/doob_common.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/doob_gain.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/doob_galerkin.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/doob_screen.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/doob_screen2.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/e2e.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/final3.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/fit_nc.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/fit_nc2.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/galerkin_xfit.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/gate_controlled.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/gate_generator.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/mcmc_pilot.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/memo5.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/ncladder.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/probe_dk.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/tscan.py
audit/2026-08-10/recovered_ephemeral/tmp_2026-08-09_10/twisted_cloning.py
audit/2026-08-10/scripts/inspect_aggregates.py
audit/2026-08-10/scripts/reproduce_amplitude.py
pps_qj/backward_pass_sector.py
scripts/aggregate.py
scripts/make_benchmark_figures.py
scripts/run_exact_benchmark.py
scripts/run_sweep_l4.py
scripts/validate_cloning.py
scripts/validate_jump_distribution.py
slurm/submit_nu_zeta1.sh
tests/conftest.py
tests/test_backward_pass_sector.py
tests/test_exact_benchmark.py
theory/VARIANCE_REDUCTION.md
```

## Untracked .bak_* module copies (redundant, but currently only copies)
```
pps_qj/cloning.py.bak_guided
pps_qj/cloning.py.bak_prelowrank
pps_qj/cloning.py.bak_spawn
pps_qj/cloning_caseA.py.bak_guided
pps_qj/cloning_caseA.py.bak_spawn
pps_qj/gaussian_backend.py.bak_guided
pps_qj/gaussian_backend.py.bak_prelowrank
pps_qj/gaussian_backend.py.bak_prenewton
pps_qj/gaussian_backend_caseA.py.bak_guided
pps_qj/parallel/grid_pps.py.bak_guidedgrid
pps_qj/parallel/worker_caseA.py.bak_guided
pps_qj/parallel/worker_clone_pps.py.bak_guided
pps_qj/parallel/worker_clone_pps.py.bak_prelowrank
theory/HANDOFF.md.bak_20260617
theory/archive/HANDOFF.md.bak_20260607_ladder
```
