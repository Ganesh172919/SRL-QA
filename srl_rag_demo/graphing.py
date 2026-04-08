"""NetworkX and Plotly graph explanation utilities."""

from __future__ import annotations

from typing import Any

import networkx as nx
import plotly.graph_objects as go

try:
    from .data_models import QAResult, RetrievalHit
except ImportError:  # pragma: no cover
    from data_models import QAResult, RetrievalHit


NODE_COLORS = {
    "question": "#2563eb",
    "document": "#64748b",
    "predicate": "#f97316",
    "role": "#16a34a",
    "frame": "#9333ea",
    "candidate": "#0f766e",
    "answer": "#dc2626",
}


def build_reasoning_graph(question: str, hits: list[RetrievalHit], result: QAResult) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("question", label="Question", type="question", detail=question)

    for hit in hits[:5]:
        doc_node = f"doc:{hit.document.doc_id}"
        graph.add_node(
            doc_node,
            label=f"Doc {hit.rank}",
            type="document",
            detail=hit.document.context,
            score=round(hit.score, 4),
        )
        graph.add_edge("question", doc_node, label=f"retrieves {hit.score:.3f}", weight=hit.score)

        predicate = hit.document.predicate or hit.document.predicate_lemma or "event"
        predicate_node = f"predicate:{hit.document.doc_id}"
        graph.add_node(predicate_node, label=predicate, type="predicate", detail=hit.document.roleset_id)
        graph.add_edge(doc_node, predicate_node, label="has predicate")

        if hit.document.frame_hint:
            frame_node = f"frame:{hit.document.doc_id}"
            graph.add_node(frame_node, label=hit.document.roleset_id or "PropBank frame", type="frame", detail=hit.document.frame_hint)
            graph.add_edge(predicate_node, frame_node, label="frame hint")

        for argument in hit.document.arguments[:8]:
            role_node = f"role:{hit.document.doc_id}:{argument.role}:{argument.start_token}:{argument.end_token}"
            graph.add_node(role_node, label=argument.role, type="role", detail=argument.text)
            graph.add_edge(predicate_node, role_node, label=argument.text[:40])

    for index, candidate in enumerate(result.candidates[:5], start=1):
        candidate_node = f"candidate:{index}"
        graph.add_node(candidate_node, label=f"{candidate.role}: {candidate.text}", type="candidate", detail="; ".join(candidate.reasons))
        doc_node = f"doc:{candidate.source_doc_id}"
        if doc_node in graph:
            graph.add_edge(doc_node, candidate_node, label=f"candidate {candidate.confidence:.3f}")
        graph.add_edge("question", candidate_node, label="answers role")

    answer_node = "answer"
    graph.add_node(answer_node, label=result.answer or "No answer", type="answer", detail="; ".join(result.reasoning))
    if result.candidates:
        graph.add_edge("candidate:1", answer_node, label=f"selected {result.confidence:.3f}")
    else:
        graph.add_edge("question", answer_node, label="no span")
    return graph


def graph_to_json(graph: nx.DiGraph) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": node, **data}
            for node, data in graph.nodes(data=True)
        ],
        "edges": [
            {"source": source, "target": target, **data}
            for source, target, data in graph.edges(data=True)
        ],
    }


def graph_to_plotly(graph: nx.DiGraph) -> go.Figure:
    if not graph.nodes:
        return go.Figure()

    positions = nx.spring_layout(graph, seed=42, k=0.9)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line={"width": 1.2, "color": "#94a3b8"},
        hoverinfo="none",
        mode="lines",
    )

    node_x: list[float] = []
    node_y: list[float] = []
    labels: list[str] = []
    colors: list[str] = []
    hover: list[str] = []
    for node, data in graph.nodes(data=True):
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(str(data.get("label", node))[:42])
        node_type = str(data.get("type", "document"))
        colors.append(NODE_COLORS.get(node_type, "#334155"))
        hover.append(f"{data.get('type', '')}: {data.get('detail', '')}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker={"size": 18, "color": colors, "line": {"width": 1, "color": "#f8fafc"}},
    )

    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin={"b": 10, "l": 10, "r": 10, "t": 10},
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            height=560,
        ),
    )
