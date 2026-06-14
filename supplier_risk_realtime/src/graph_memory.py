from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import networkx as nx

from .config import RiskEvent, SupplierProfile


class GraphMemory:
    # Graph memory lets the system learn repeated entities and event clusters.
    def __init__(self, base_dir: str = "shared", supplier_name: str = "default") -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / f"{supplier_name}_graph.json"

    def load(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        if not self.path.exists():
            return g
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            g = nx.node_link_graph(data, directed=True, multigraph=True)
        except Exception:
            g = nx.MultiDiGraph()
        return g

    def save(self, graph: nx.MultiDiGraph) -> None:
        data = nx.node_link_data(graph)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def update(self, profile: SupplierProfile, events: Iterable[RiskEvent]) -> nx.MultiDiGraph:
        g = self.load()

        g.add_node(profile.name, kind="supplier", country=profile.hq_country, ticker=profile.ticker)
        g.add_node(profile.hq_country, kind="country")

        for ev in events:
            event_node = f"{ev.category}:{ev.title}"
            g.add_node(
                event_node,
                kind="event",
                category=ev.category,
                severity=float(ev.severity),
                source=ev.source,
            )
            g.add_edge(profile.name, event_node, relation="observed")
            if profile.hq_country:
                g.add_edge(profile.hq_country, event_node, relation="country_exposure")
            for ent in ev.entities:
                if ent:
                    g.add_node(ent, kind="entity")
                    g.add_edge(ent, event_node, relation="mentioned_in")

        self.save(g)
        return g

    def summary(self, top_k: int = 8) -> list[str]:
        # Return the most connected nodes, which usually show the active risk clusters.
        g = self.load()
        if g.number_of_nodes() == 0:
            return ["graph empty"]
        degree_rank = sorted(g.degree(), key=lambda x: x[1], reverse=True)
        out = []
        for node, deg in degree_rank[:top_k]:
            kind = g.nodes[node].get("kind", "node")
            out.append(f"{node} | {kind} | degree={deg}")
        return out
