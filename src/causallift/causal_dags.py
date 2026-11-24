"""Classical backdoor identification in explicitly supplied DAGs."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx


@dataclass(frozen=True)
class CausalDAG:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    treatment: str
    outcome: str
    observed: tuple[str, ...] = ()
    latent: tuple[str, ...] = ()
    adjustment: tuple[str, ...] = ()
    assumptions: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("nodes", "observed", "latent", "adjustment"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "edges", tuple(map(tuple, self.edges)))
        nodes = set(self.nodes)
        endpoints = {self.treatment, self.outcome}
        if len(nodes) != len(self.nodes) or len(endpoints) != 2 or not endpoints <= nodes:
            raise ValueError("Use unique nodes and distinct treatment and outcome nodes")
        if any(len(edge) != 2 or not set(edge) <= nodes for edge in self.edges):
            raise ValueError("Edges must be directed pairs of known nodes")
        observed, latent = set(self.observed), set(self.latent)
        if observed & latent or (observed | latent) != nodes - endpoints:
            raise ValueError("Classify each non-endpoint node as observed or latent")
        if not set(self.adjustment) <= observed:
            raise ValueError("Default adjustment must contain only observed covariates")
        if not nx.is_directed_acyclic_graph(self.graph()):
            raise ValueError("Causal graphs must be acyclic")

    def graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)
        return graph

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def backdoor(self, adjustment=None) -> dict:
        selected = set(self.adjustment if adjustment is None else adjustment)
        graph = self.graph()
        invalid = (selected - set(self.observed)) | (selected & nx.descendants(graph, self.treatment))
        graph.remove_edges_from(list(graph.out_edges(self.treatment)))
        # Report separation even for observed descendants, but never accept them.
        admissible_for_separation = selected <= set(self.observed)
        separated = (nx.is_d_separator(graph, {self.treatment}, {self.outcome}, selected)
                     if admissible_for_separation else None)
        identified = bool(separated and not invalid)
        warnings = []
        if invalid:
            warnings.append("Classical backdoor adjustment requires observed non-descendants of treatment.")
        if separated is False:
            warnings.append("The proposed set leaves a path open in the backdoor graph.")
        return {"estimand": "total average effect of treatment on outcome", "adjustment_set": sorted(selected),
                "identified": identified, "backdoor_d_separated": separated, "invalid": sorted(invalid),
                "warnings": warnings, "assumptions": {"graph_is_correct": True, "consistency": True,
                                                     "positivity_for_adjustment": True, **self.assumptions}}
