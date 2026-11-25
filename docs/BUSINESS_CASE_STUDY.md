# Business case study

## Synthetic benchmark

Scenario E uses a contribution of 45 per incremental conversion and a fixed treatment cost of 2. At the 1,600 ceiling, all four targeting policies spent 1,600. Profits are in simulated units.

| Policy | Expected true profit | DM | IPS | DR |
| --- | ---: | ---: | ---: | ---: |
| Random | -956 | -591 | -520 | 1,088 |
| Predictive | 1,930 | 815 | 2,450 | 2,876 |
| Uplift | 4,160 | 2,385 | 4,160 | 4,447 |
| Profit-aware | 4,160 | 2,385 | 4,160 | 4,447 |

Profit-aware targeting matched uplift under uniform economics and exceeded predictive targeting by 2,230 in expected truth. The paired DR estimate was 1,571 with a conditional 95% interval of [-1,629, 4,770]. DR reversed the sign for random targeting in this validation cohort. Profit-aware expected profit increased from 1,040 to 2,080 to 4,160 at ceilings of 400, 800 and 1,600. Holding contribution at 45 and the ceiling at 1,600, unit costs of 1, 2 and 4 gave expected profits of 7,440, 4,160 and 1,280 after reallocation. At unit cost 1, capacity bound the policy at 1,200 treatments and 1,200 spend.

With variable contributions and the 1,600 ceiling, profit ranking returned 5,621 in expected truth versus 4,095 for uplift and 6,024 for the oracle. With variable costs and a 72 ceiling on 180 rows, greedy profit-to-cost returned 185.16, the learned integer allocation returned 179.68, and the oracle returned 193.44. The integer solver maximized predicted profit. Under unfavorable economics, profit ranking selected nobody, while predictive, random and uplift policies had negative expected profit.

## Criteo released 1M validation benchmark

The cohort had 583 conversions, including 58 controls. The 20% nominal capacity corresponds to 40,000 treatments and a 16,000 ceiling. Monetary values below use hypothetical economics.

| Policy | Selected | Spend | DR incremental conversions | DR profit | Conditional 95% profit interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Random | 40,000 | 16,000 | 26 | -13,653 | [-18,533, -8,772] |
| Predictive | 40,000 | 16,000 | 153 | -902 | [-11,522, 9,718] |
| Uplift | 40,000 | 16,000 | 161 | -205 | [-10,247, 9,836] |
| Profit-aware | 6,514 | 2,606 | 97 | 6,189 | [-2,954, 15,333] |

The profit-aware policy had a paired DR advantage of 7,092 over predictive targeting, with a conditional 95% interval of [1,690, 12,493]. Those policies shared a ceiling and capacity but used different spending. The estimated profit relative to treat-none remains uncertain. The released sample and unknown original trial propensities limit causal interpretation. Scenario F remains research-only.
