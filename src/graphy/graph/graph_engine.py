"""High-performance Directed Weighted Prerequisite Graph Engine."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from graphy.graph.models import GraphVertex, PrerequisiteEdge


class KnowledgeDiGraph:
    """Directed weighted prerequisite graph supporting CSR/CSC analytics, cycle breaking, and mastery subgraphs."""

    def __init__(self, book_id: str) -> None:
        self.book_id = book_id
        self._vertices: Dict[str, GraphVertex] = {}
        self._out_edges: Dict[str, Dict[str, PrerequisiteEdge]] = {}
        self._in_edges: Dict[str, Dict[str, PrerequisiteEdge]] = {}

    def add_vertex(self, vertex: GraphVertex) -> None:
        self._vertices[vertex.chunk_id] = vertex
        self._out_edges.setdefault(vertex.chunk_id, {})
        self._in_edges.setdefault(vertex.chunk_id, {})

    def add_edge(self, edge: PrerequisiteEdge) -> None:
        if edge.source_id == edge.target_id:
            return
        if edge.source_id not in self._vertices or edge.target_id not in self._vertices:
            raise KeyError(f"Both source ({edge.source_id}) and target ({edge.target_id}) must exist in graph.")
        self._out_edges[edge.source_id][edge.target_id] = edge
        self._in_edges[edge.target_id][edge.source_id] = edge

    def remove_edge(self, source_id: str, target_id: str) -> None:
        self._out_edges.get(source_id, {}).pop(target_id, None)
        self._in_edges.get(target_id, {}).pop(source_id, None)

    def get_vertex(self, chunk_id: str) -> Optional[GraphVertex]:
        return self._vertices.get(chunk_id)

    def vertices(self) -> List[GraphVertex]:
        return sorted(self._vertices.values(), key=lambda v: v.vertex_index)

    def edges(self) -> List[PrerequisiteEdge]:
        all_edges: List[PrerequisiteEdge] = []
        for src_id in sorted(self._out_edges.keys()):
            for dst_id in sorted(self._out_edges[src_id].keys()):
                all_edges.append(self._out_edges[src_id][dst_id])
        all_edges.sort(key=lambda e: (e.source_index, e.target_index))
        return all_edges

    def outgoing_edges(self, chunk_id: str) -> List[PrerequisiteEdge]:
        return list(self._out_edges.get(chunk_id, {}).values())

    def incoming_edges(self, chunk_id: str) -> List[PrerequisiteEdge]:
        return list(self._in_edges.get(chunk_id, {}).values())

    def outgoing_weight_sum(self, chunk_id: str, excluded_nodes: Optional[Set[str]] = None) -> float:
        excl = excluded_nodes or set()
        if chunk_id in excl:
            return 0.0
        return round(
            sum(
                e.similarity_weight
                for dst, e in self._out_edges.get(chunk_id, {}).items()
                if dst not in excl
            ),
            6,
        )

    def incoming_weight_sum(self, chunk_id: str, excluded_nodes: Optional[Set[str]] = None) -> float:
        excl = excluded_nodes or set()
        if chunk_id in excl:
            return 0.0
        return round(
            sum(
                e.similarity_weight
                for src, e in self._in_edges.get(chunk_id, {}).items()
                if src not in excl
            ),
            6,
        )

    def break_cycles(self) -> int:
        """Detect and break directed cycles by removing the weakest/backward edge in each cycle."""
        cycles_broken = 0
        while True:
            cycle_edges = self._find_one_cycle()
            if not cycle_edges:
                break
            # Pick the weakest edge in the cycle (lowest similarity_weight * llm_confidence,
            # breaking ties by removing backward-in-textbook edges first)
            weakest = min(
                cycle_edges,
                key=lambda e: (
                    round(e.similarity_weight * e.llm_confidence, 6),
                    1 if e.source_index < e.target_index else 0,
                ),
            )
            self.remove_edge(weakest.source_id, weakest.target_id)
            cycles_broken += 1
        return cycles_broken

    def _find_one_cycle(self) -> Optional[List[PrerequisiteEdge]]:
        """DFS cycle detector returning the list of PrerequisiteEdges forming a cycle if one exists."""
        visited: Set[str] = set()
        on_stack: Set[str] = set()
        parent_edge: Dict[str, PrerequisiteEdge] = {}

        for start_v in sorted(self._vertices.keys()):
            if start_v in visited:
                continue

            stack: List[Tuple[str, int, List[PrerequisiteEdge]]] = [
                (start_v, 0, self.outgoing_edges(start_v))
            ]
            visited.add(start_v)
            on_stack.add(start_v)

            while stack:
                curr_u, idx, out_list = stack[-1]
                if idx < len(out_list):
                    edge = out_list[idx]
                    stack[-1] = (curr_u, idx + 1, out_list)
                    nxt_v = edge.target_id

                    if nxt_v in on_stack:
                        # Reconstruct cycle
                        cycle: List[PrerequisiteEdge] = [edge]
                        trace = curr_u
                        while trace != nxt_v and trace in parent_edge:
                            pe = parent_edge[trace]
                            cycle.append(pe)
                            trace = pe.source_id
                        return cycle

                    if nxt_v not in visited:
                        visited.add(nxt_v)
                        on_stack.add(nxt_v)
                        parent_edge[nxt_v] = edge
                        stack.append((nxt_v, 0, self.outgoing_edges(nxt_v)))
                else:
                    on_stack.remove(curr_u)
                    stack.pop()

        return None
