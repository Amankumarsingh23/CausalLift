# Causal DAGs

Each JSON graph declares nodes, ordered cause-to-effect edges, observed and latent covariates, treatment, outcome, optional default adjustment, and assumption metadata. `CausalDAG.load` and `save` support a single graph definition. The report bundles the four examples by name.

For the total effect, classical backdoor adjustment requires observed variables that are not treatment descendants. Delete all outgoing treatment edges and use NetworkX d-separation on the resulting graph. A set is accepted only if it passes both checks. The collider is rejected as a treatment descendant. A separate demonstration removes only the direct treatment-to-outcome edge to isolate the path opened by conditioning on the collider. This checker supports the classical backdoor criterion. It neither discovers graphs from data nor decides identification by other criteria. [NetworkX d-separation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.d_separation.is_d_separator.html)

- A confounder causes both assignment and outcome. Here adjusting for X blocks the backdoor path.
- A mediator transmits treatment effects. In the additive fixture its adjustment moves the coefficient toward the direct effect of 1 instead of the total effect of 4. This interpretation is specific to the simulated equations.
- A collider is caused by two variables on a path. Conditioning on C associates randomized treatment with the hidden cause U of outcome and opens a noncausal path.
- In the hidden-confounding graph, U causes treatment and outcome. No available adjustment blocks that path.

Graph-based identification assumes the graph is correct. Estimation also needs conditional exchangeability as warranted by the graph, consistency between observed and assigned potential outcomes, and positivity in the target population. None follows from visually overlapping score distributions.
