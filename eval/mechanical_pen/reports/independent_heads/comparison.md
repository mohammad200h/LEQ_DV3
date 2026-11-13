### new10hz (seeds: chained 3, independent 3)

| Group | Metric | Chained (old) | Independent (fixed) | Change | Verdict |
|---|---|---|---|---|---|
| World model | Forces recon loss, held-out | 4.100 ± 0.011 | 4.015 ± 0.100 | -2% | within seed spread |
| World model | Positions recon loss, held-out | 0.00296 ± 0.00027 | 0.00281 ± 0.00014 | -5% | within seed spread |
| World model | Open-loop forces loss, held-out | 8.074 ± 0.419 | 7.665 ± 0.155 | -5% | within seed spread |
| World model | KL posterior-prior, held-out (nats) | 6.07 ± 0.30 | 4.97 ± 0.20 | -18% | better |
| World model | Prior classes/var, held-out | 5.53 ± 0.14 | 5.54 ± 0.02 | +0% | within seed spread |
| World model | Position error, expert actions (mm) | 1.31 ± 0.07 | 1.34 ± 0.07 | +2% | within seed spread |
| World model | Force RMSE, expert actions (counts) | 24.0 ± 0.5 | 20.1 ± 3.0 | -16% | better |
| World model | Hand motion, expert actions (x real) | 1.74 ± 0.02 | 1.69 ± 0.02 | -3% | better |
| Ensemble | Mean head open-loop force RMSE (counts) | 31.4 ± 0.3 | 30.4 ± 0.8 | -3% | within seed spread |
| Ensemble | Worst / best head force MSE | 1.02 ± 0.01 | 1.02 ± 0.00 | -0% | within seed spread |
| Ensemble | Head spread / error, forces | 0.117 ± 0.006 | 0.135 ± 0.010 | +16% | better |
| Ensemble | Head spread / error, positions | 0.152 ± 0.006 | 0.163 ± 0.003 | +7% | better |
| Ensemble | Log force MSE per head index | +0.0018 ± 0.0028 | +0.0005 ± 0.0019 | -75% | within seed spread |
| DITTO | DITTO reward, held-out | 0.956 ± 0.004 | 0.952 ± 0.001 | -0% | within seed spread |
| DITTO | DITTO action MSE, held-out | 0.0040 ± 0.0011 | 0.0010 ± 0.0001 | -75% | better |
| DITTO | Policy-on-posterior command error (rad) | 0.123 ± 0.016 | 0.049 ± 0.004 | -60% | better |
| DITTO | Policy-in-prior command error (rad) | 0.161 ± 0.009 | 0.098 ± 0.008 | -39% | better |
| DITTO | Policy-in-prior position error (mm) | 4.36 ± 0.06 | 3.84 ± 0.42 | -12% | better |
| DITTO | Policy jitter on posterior (x expert) | 5.52 ± 0.11 | 5.08 ± 0.10 | -8% | better |
| DITTO | Policy jitter in prior (x expert) | 5.90 ± 0.24 | 5.00 ± 0.15 | -15% | better |

### new50hz (seeds: chained 3, independent 3)

| Group | Metric | Chained (old) | Independent (fixed) | Change | Verdict |
|---|---|---|---|---|---|
| World model | Forces recon loss, held-out | 3.843 ± 0.103 | 3.843 ± 0.058 | -0% | within seed spread |
| World model | Positions recon loss, held-out | 0.00295 ± 0.00046 | 0.00276 ± 0.00014 | -6% | within seed spread |
| World model | Open-loop forces loss, held-out | 5.365 ± 0.249 | 4.955 ± 0.114 | -8% | better |
| World model | KL posterior-prior, held-out (nats) | 2.95 ± 0.07 | 2.67 ± 0.00 | -9% | better |
| World model | Prior classes/var, held-out | 6.51 ± 0.13 | 6.06 ± 0.02 | -7% | lower |
| World model | Position error, expert actions (mm) | 1.34 ± 0.05 | 1.46 ± 0.08 | +9% | within seed spread |
| World model | Force RMSE, expert actions (counts) | 25.7 ± 2.8 | 27.8 ± 5.5 | +8% | within seed spread |
| World model | Hand motion, expert actions (x real) | 4.92 ± 0.21 | 5.10 ± 0.24 | +4% | within seed spread |
| Ensemble | Mean head open-loop force RMSE (counts) | 23.9 ± 0.4 | 23.0 ± 0.5 | -4% | better |
| Ensemble | Worst / best head force MSE | 1.02 ± 0.01 | 1.03 ± 0.01 | +1% | within seed spread |
| Ensemble | Head spread / error, forces | 0.163 ± 0.005 | 0.191 ± 0.007 | +17% | better |
| Ensemble | Head spread / error, positions | 0.166 ± 0.006 | 0.175 ± 0.001 | +5% | better |
| Ensemble | Log force MSE per head index | +0.0011 ± 0.0018 | -0.0015 ± 0.0032 | -243% | within seed spread |
| DITTO | DITTO reward, held-out | 0.986 ± 0.000 | 0.987 ± 0.001 | +0% | within seed spread |
| DITTO | DITTO action MSE, held-out | 0.0053 ± 0.0011 | 0.0012 ± 0.0006 | -78% | better |
| DITTO | Policy-on-posterior command error (rad) | 0.114 ± 0.010 | 0.059 ± 0.009 | -49% | better |
| DITTO | Policy-in-prior command error (rad) | 0.255 ± 0.036 | 0.208 ± 0.036 | -18% | within seed spread |
| DITTO | Policy-in-prior position error (mm) | 8.48 ± 1.45 | 8.27 ± 1.24 | -3% | within seed spread |
| DITTO | Policy jitter on posterior (x expert) | 28.84 ± 1.33 | 25.68 ± 1.80 | -11% | better |
| DITTO | Policy jitter in prior (x expert) | 31.88 ± 2.01 | 27.85 ± 2.20 | -13% | within seed spread |

### new100hz (seeds: chained 3, independent 3)

| Group | Metric | Chained (old) | Independent (fixed) | Change | Verdict |
|---|---|---|---|---|---|
| World model | Forces recon loss, held-out | 4.066 ± 0.033 | 3.894 ± 0.064 | -4% | better |
| World model | Positions recon loss, held-out | 0.00305 ± 0.00012 | 0.00317 ± 0.00015 | +4% | within seed spread |
| World model | Open-loop forces loss, held-out | 5.013 ± 0.045 | 4.610 ± 0.102 | -8% | better |
| World model | KL posterior-prior, held-out (nats) | 2.62 ± 0.06 | 2.49 ± 0.02 | -5% | better |
| World model | Prior classes/var, held-out | 6.37 ± 0.12 | 6.19 ± 0.03 | -3% | lower |
| World model | Position error, expert actions (mm) | 1.47 ± 0.15 | 1.53 ± 0.22 | +4% | within seed spread |
| World model | Force RMSE, expert actions (counts) | 30.2 ± 5.3 | 35.8 ± 3.7 | +19% | within seed spread |
| World model | Hand motion, expert actions (x real) | 9.74 ± 0.40 | 10.12 ± 0.76 | +4% | within seed spread |
| Ensemble | Mean head open-loop force RMSE (counts) | 23.6 ± 0.4 | 22.9 ± 0.2 | -3% | better |
| Ensemble | Worst / best head force MSE | 1.02 ± 0.00 | 1.02 ± 0.01 | -0% | within seed spread |
| Ensemble | Head spread / error, forces | 0.176 ± 0.004 | 0.192 ± 0.006 | +9% | better |
| Ensemble | Head spread / error, positions | 0.172 ± 0.003 | 0.175 ± 0.006 | +2% | within seed spread |
| Ensemble | Log force MSE per head index | +0.0002 ± 0.0019 | +0.0009 ± 0.0012 | +318% | within seed spread |
| DITTO | DITTO reward, held-out | 0.991 ± 0.000 | 0.991 ± 0.000 | +0% | within seed spread |
| DITTO | DITTO action MSE, held-out | 0.0061 ± 0.0010 | 0.0037 ± 0.0001 | -39% | better |
| DITTO | Policy-on-posterior command error (rad) | 0.130 ± 0.006 | 0.081 ± 0.013 | -38% | better |
| DITTO | Policy-in-prior command error (rad) | 0.314 ± 0.039 | 0.203 ± 0.033 | -35% | better |
| DITTO | Policy-in-prior position error (mm) | 10.28 ± 1.56 | 7.04 ± 1.91 | -32% | within seed spread |
| DITTO | Policy jitter on posterior (x expert) | 56.37 ± 1.03 | 53.91 ± 2.99 | -4% | within seed spread |
| DITTO | Policy jitter in prior (x expert) | 65.15 ± 7.36 | 58.78 ± 2.33 | -10% | within seed spread |
