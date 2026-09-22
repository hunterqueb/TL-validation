# Standalone rollout results

reference = SGP4 from space-track TLEs. Hand-offs measure consistency, not accuracy.

Protocol: `4a25bddbb77a0ca03c7ce84498d1c07f6fccef4ec754ef75792130ba5454a437`. Three development seeds on the first orbit of window 0; later-window models each observe only their own first orbit. CPU float64. The forecast spans five orbits with no observations.

Selected optimizer steps: `{"lstm_one_step": 750, "lstm_rollout": 100, "mamba_one_step": 850, "mamba_rollout": 300}`.

Zero selected steps means the learned correction was not selected over physics. Budgets are not compute-matched; per-run costs are saved.

[Observation budget](figures/4a25bddbb77a/observation-budget.png) · [Validation curves](figures/4a25bddbb77a/validation.png)

| Phase | Window | Method | Successful / expected | Position RMSE median ± IQR (km) | RMSE / physics |
|---|---:|---|---:|---:|---:|
| walkthrough | 0 | persistence | 1/1 | 59610.2002 ± 0.0000 | 1332.7413 |
| walkthrough | 0 | two_body | 1/1 | 44.7275 ± 0.0000 | 1.0000 |
| walkthrough | 0 | mamba_one_step | 3/3 | 217.7261 ± 3.8319 | 4.8678 |
| walkthrough | 0 | mamba_rollout | 3/3 | 1432.6872 ± 2.2834 | 32.0315 |
| walkthrough | 0 | lstm_one_step | 3/3 | 217.7028 ± 1.5398 | 4.8673 |
| walkthrough | 0 | lstm_rollout | 3/3 | 1424.5016 ± 5.0408 | 31.8484 |
| evaluate | 4 | persistence | 1/1 | 59612.0535 ± 0.0000 | 1307.0417 |
| evaluate | 4 | two_body | 1/1 | 45.6084 ± 0.0000 | 1.0000 |
| evaluate | 4 | mamba_one_step | 3/3 | 222.1350 ± 1.2474 | 4.8705 |
| evaluate | 4 | mamba_rollout | 3/3 | 1458.3879 ± 0.5863 | 31.9763 |
| evaluate | 4 | lstm_one_step | 3/3 | 222.1779 ± 2.1280 | 4.8714 |
| evaluate | 4 | lstm_rollout | 3/3 | 1453.6875 ± 1.7243 | 31.8733 |
| evaluate | 5 | persistence | 1/1 | 59612.3328 ± 0.0000 | 1289.9987 |
| evaluate | 5 | two_body | 1/1 | 46.2112 ± 0.0000 | 1.0000 |
| evaluate | 5 | mamba_one_step | 3/3 | 160.3269 ± 3.6242 | 3.4694 |
| evaluate | 5 | mamba_rollout | 3/3 | 1027.3984 ± 5.7486 | 22.2327 |
| evaluate | 5 | lstm_one_step | 3/3 | 158.9750 ± 1.9728 | 3.4402 |
| evaluate | 5 | lstm_rollout | 3/3 | 1016.6008 ± 9.6931 | 21.9990 |
| evaluate | 6 | persistence | 1/1 | 59613.1935 ± 0.0000 | 1569.5609 |
| evaluate | 6 | two_body | 1/1 | 37.9808 ± 0.0000 | 1.0000 |
| evaluate | 6 | mamba_one_step | 3/3 | 142.2018 ± 6.5797 | 3.7440 |
| evaluate | 6 | mamba_rollout | 3/3 | 812.9478 ± 2.0987 | 21.4042 |
| evaluate | 6 | lstm_one_step | 3/3 | 133.6291 ± 1.4637 | 3.5183 |
| evaluate | 6 | lstm_rollout | 3/3 | 818.5767 ± 6.3711 | 21.5524 |
| evaluate | 7 | persistence | 1/1 | 59613.8012 ± 0.0000 | 1330.1265 |
| evaluate | 7 | two_body | 1/1 | 44.8181 ± 0.0000 | 1.0000 |
| evaluate | 7 | mamba_one_step | 3/3 | 233.3418 ± 2.2541 | 5.2064 |
| evaluate | 7 | mamba_rollout | 3/3 | 1538.9220 ± 2.2989 | 34.3370 |
| evaluate | 7 | lstm_one_step | 3/3 | 233.5909 ± 1.1393 | 5.2120 |
| evaluate | 7 | lstm_rollout | 3/3 | 1526.8384 ± 1.9902 | 34.0674 |

## Trajectory figures

- [Trajectory diagnostics, walkthrough window 0](figures/4a25bddbb77a/walkthrough-window-0.png)
- [Trajectory diagnostics, evaluate window 4](figures/4a25bddbb77a/evaluate-window-4.png)
- [Trajectory diagnostics, evaluate window 5](figures/4a25bddbb77a/evaluate-window-5.png)
- [Trajectory diagnostics, evaluate window 6](figures/4a25bddbb77a/evaluate-window-6.png)
- [Trajectory diagnostics, evaluate window 7](figures/4a25bddbb77a/evaluate-window-7.png)

## Paired rollout / one-step RMSE ratios

Ratios below one favor rollout training. Each pair uses the same window and seed.

- walkthrough window 0, mamba_rollout, seed 0: 6.5581
- walkthrough window 0, mamba_rollout, seed 1: 6.6004
- walkthrough window 0, mamba_rollout, seed 2: 6.7958
- walkthrough window 0, lstm_rollout, seed 0: 6.4864
- walkthrough window 0, lstm_rollout, seed 1: 6.5319
- walkthrough window 0, lstm_rollout, seed 2: 6.5433
- evaluate window 6, mamba_rollout, seed 0: 5.6327
- evaluate window 4, mamba_rollout, seed 0: 6.5983
- evaluate window 7, mamba_rollout, seed 0: 6.5107
- evaluate window 5, mamba_rollout, seed 0: 6.3170
- evaluate window 6, mamba_rollout, seed 1: 6.1644
- evaluate window 5, mamba_rollout, seed 1: 6.4523
- evaluate window 7, mamba_rollout, seed 1: 6.6082
- evaluate window 4, mamba_rollout, seed 1: 6.5702
- evaluate window 6, mamba_rollout, seed 2: 5.7169
- evaluate window 4, mamba_rollout, seed 2: 6.5251
- evaluate window 5, mamba_rollout, seed 2: 6.5833
- evaluate window 7, mamba_rollout, seed 2: 6.6441
- evaluate window 5, lstm_rollout, seed 0: 6.4061
- evaluate window 7, lstm_rollout, seed 0: 6.5206
- evaluate window 6, lstm_rollout, seed 0: 6.0818
- evaluate window 4, lstm_rollout, seed 0: 6.4929
- evaluate window 7, lstm_rollout, seed 1: 6.5364
- evaluate window 6, lstm_rollout, seed 1: 6.0445
- evaluate window 5, lstm_rollout, seed 1: 6.4436
- evaluate window 4, lstm_rollout, seed 1: 6.5306
- evaluate window 5, lstm_rollout, seed 2: 6.3947
- evaluate window 6, lstm_rollout, seed 2: 6.1257
- evaluate window 4, lstm_rollout, seed 2: 6.6221
- evaluate window 7, lstm_rollout, seed 2: 6.5674

## Across later windows

These summaries use the four window-level medians, not pooled seed rows.

- mamba_one_step: median window RMSE 191.2310 km; range 142.2018–233.3418; lower than physics on 0/4 windows.
- mamba_rollout: median window RMSE 1242.8931 km; range 812.9478–1538.9220; lower than physics on 0/4 windows.
- lstm_one_step: median window RMSE 190.5765 km; range 133.6291–233.5909; lower than physics on 0/4 windows.
- lstm_rollout: median window RMSE 1235.1441 km; range 818.5767–1526.8384; lower than physics on 0/4 windows.

## Provenance and limitations

One object, SGP4-derived reference, additional window-0 development data. Correction is constant within each RK4 observation step and bounded per component. Energy/angular momentum drift are diagnostics, not targets for perfect conservation. All metric and hand-off details, actual endpoint times, predictions, costs, and failures are in the run directories below. No original-demo result rows were mixed in.

Development runs: 20260921T175030-f47e799b, 20260921T175031-1b63c345, 20260921T175030-8a0a656a, 20260921T175040-e9c40846, 20260921T175040-98c311a6, 20260921T175040-32e21726, 20260921T175327-5d955cca, 20260921T175326-e48b3080, 20260921T175329-9558e0bd, 20260921T175334-66c4f702, 20260921T175332-798a4a37, 20260921T175336-c67a38b7

All evaluation attempts:

- `20260921T175703-842201f8`: walkthrough persistence w0 sNone — ok
- `20260921T175704-c567afae`: walkthrough two_body w0 sNone — ok
- `20260921T175705-a99c6c4b`: walkthrough mamba_one_step w0 s0 — ok
- `20260921T175712-953ad2ea`: walkthrough mamba_one_step w0 s1 — ok
- `20260921T175719-2fb1b13d`: walkthrough mamba_one_step w0 s2 — ok
- `20260921T175726-5218d01b`: walkthrough mamba_rollout w0 s0 — ok
- `20260921T175735-97f7148f`: walkthrough mamba_rollout w0 s1 — ok
- `20260921T175745-5ba5c65c`: walkthrough mamba_rollout w0 s2 — ok
- `20260921T175754-7fcb89f2`: walkthrough lstm_one_step w0 s0 — ok
- `20260921T175759-3f5a86d0`: walkthrough lstm_one_step w0 s1 — ok
- `20260921T175803-f3b9d13b`: walkthrough lstm_one_step w0 s2 — ok
- `20260921T175807-c59decf5`: walkthrough lstm_rollout w0 s0 — ok
- `20260921T175812-7eed5c1c`: walkthrough lstm_rollout w0 s1 — ok
- `20260921T175816-b30bc4dd`: walkthrough lstm_rollout w0 s2 — ok
- `20260921T175855-549324ac`: evaluate persistence w6 sNone — ok
- `20260921T175856-00e02638`: evaluate two_body w6 sNone — ok
- `20260921T175856-2322ad70`: evaluate persistence w7 sNone — ok
- `20260921T175856-5c19fe71`: evaluate two_body w4 sNone — ok
- `20260921T175856-632719da`: evaluate persistence w5 sNone — ok
- `20260921T175856-8458687c`: evaluate two_body w7 sNone — ok
- `20260921T175856-9f213cb3`: evaluate persistence w4 sNone — ok
- `20260921T175856-c5661e5d`: evaluate two_body w5 sNone — ok
- `20260921T175857-0428a3a1`: evaluate mamba_one_step w6 s0 — ok
- `20260921T175857-6489a4c8`: evaluate mamba_one_step w4 s0 — ok
- `20260921T175858-0ae2291a`: evaluate mamba_one_step w7 s0 — ok
- `20260921T175858-57ebb7dd`: evaluate mamba_one_step w5 s0 — ok
- `20260921T175908-1e07404a`: evaluate mamba_one_step w4 s1 — ok
- `20260921T175908-a4824b50`: evaluate mamba_one_step w6 s1 — ok
- `20260921T175909-76af0d58`: evaluate mamba_one_step w7 s1 — ok
- `20260921T175909-e787fa3d`: evaluate mamba_one_step w5 s1 — ok
- `20260921T175919-ca0deb1a`: evaluate mamba_one_step w4 s2 — ok
- `20260921T175919-d2d901db`: evaluate mamba_one_step w6 s2 — ok
- `20260921T175920-733e5cbc`: evaluate mamba_one_step w5 s2 — ok
- `20260921T175920-f95c9883`: evaluate mamba_one_step w7 s2 — ok
- `20260921T175930-fe74f42c`: evaluate mamba_rollout w6 s0 — ok
- `20260921T175931-06e0074c`: evaluate mamba_rollout w4 s0 — ok
- `20260921T175931-29c00ddb`: evaluate mamba_rollout w7 s0 — ok
- `20260921T175931-f119f5fa`: evaluate mamba_rollout w5 s0 — ok
- `20260921T175946-4bb4e4e1`: evaluate mamba_rollout w6 s1 — ok
- `20260921T175947-6c478235`: evaluate mamba_rollout w5 s1 — ok
- `20260921T175947-f1f1a1bf`: evaluate mamba_rollout w7 s1 — ok
- `20260921T175947-f7cc22d2`: evaluate mamba_rollout w4 s1 — ok
- `20260921T180002-c3520983`: evaluate mamba_rollout w6 s2 — ok
- `20260921T180002-cb331358`: evaluate mamba_rollout w4 s2 — ok
- `20260921T180003-4f512c52`: evaluate mamba_rollout w5 s2 — ok
- `20260921T180003-c276e680`: evaluate mamba_rollout w7 s2 — ok
- `20260921T180018-4e9ee996`: evaluate lstm_one_step w6 s0 — ok
- `20260921T180019-a6da4050`: evaluate lstm_one_step w5 s0 — ok
- `20260921T180019-cf0b165e`: evaluate lstm_one_step w7 s0 — ok
- `20260921T180019-d774d88f`: evaluate lstm_one_step w4 s0 — ok
- `20260921T180026-5100d481`: evaluate lstm_one_step w5 s1 — ok
- `20260921T180026-ae67834a`: evaluate lstm_one_step w7 s1 — ok
- `20260921T180026-d8c68304`: evaluate lstm_one_step w4 s1 — ok
- `20260921T180026-f257ff59`: evaluate lstm_one_step w6 s1 — ok
- `20260921T180033-47d60182`: evaluate lstm_one_step w6 s2 — ok
- `20260921T180033-4c5270ce`: evaluate lstm_one_step w7 s2 — ok
- `20260921T180033-c97b9c22`: evaluate lstm_one_step w5 s2 — ok
- `20260921T180033-d66e6d3b`: evaluate lstm_one_step w4 s2 — ok
- `20260921T180040-12a5dbad`: evaluate lstm_rollout w5 s0 — ok
- `20260921T180040-7b595db2`: evaluate lstm_rollout w7 s0 — ok
- `20260921T180040-7dc4ca68`: evaluate lstm_rollout w6 s0 — ok
- `20260921T180040-876e80a2`: evaluate lstm_rollout w4 s0 — ok
- `20260921T180047-2238cfb0`: evaluate lstm_rollout w7 s1 — ok
- `20260921T180047-4d702a1e`: evaluate lstm_rollout w6 s1 — ok
- `20260921T180047-b2d4195a`: evaluate lstm_rollout w5 s1 — ok
- `20260921T180047-dfd2b7ff`: evaluate lstm_rollout w4 s1 — ok
- `20260921T180053-346364d3`: evaluate lstm_rollout w5 s2 — ok
- `20260921T180053-7a4128d8`: evaluate lstm_rollout w6 s2 — ok
- `20260921T180054-97491d83`: evaluate lstm_rollout w4 s2 — ok
- `20260921T180054-cf8583d2`: evaluate lstm_rollout w7 s2 — ok
