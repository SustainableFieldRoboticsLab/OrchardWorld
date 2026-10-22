# E2 - oracle-feature probe (privileged stem strength + GT occlusion), test split

Mean +- s.d. over 3 training seeds; same test rows as Table 2.

| model | detach_auroc | success_auroc | success_ap | contact_auroc | damage_auroc |
|---|---|---|---|---|---|
| standard | 0.778 +- 0.002 | 0.715 +- 0.005 | 0.469 +- 0.009 | 0.772 +- 0.006 | 0.682 +- 0.009 |
| oracle | 0.770 +- 0.009 | 0.729 +- 0.005 | 0.479 +- 0.007 | 0.764 +- 0.003 | 0.679 +- 0.003 |
| delta (oracle - standard) | -0.008 | +0.014 | +0.010 | -0.008 | -0.003 |

The 'standard' rows must match Table 2 (0.778 / 0.469 / 0.772); otherwise stop and report.
