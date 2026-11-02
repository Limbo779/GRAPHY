"""Apache GraphAr Standard Layout Persistence for Directed Weighted Knowledge Graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import yaml

from graphy.graph.graph_engine import KnowledgeDiGraph
from graphy.graph.models import GraphVertex, PrerequisiteEdge


class GraphArStore:
    """Reads and writes KnowledgeDiGraph in Apache GraphAr standard directory & YAML format."""

    VERTEX_CHUNK_SIZE = 1024
    EDGE_CHUNK_SIZE = 4096

    def __init__(self, book_id: str, graphs_root: Path | str = "data/graphs") -> None:
        self.book_id = book_id
        self.graph_dir = Path(graphs_root) / book_id

    def save(self, graph: KnowledgeDiGraph) -> Path:
        """Persist `graph` into GraphAr YAML descriptors and partitioned CSR/CSC vertex/edge chunks."""
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        vertices = graph.vertices()
        edges = graph.edges()

        # 1. Write root graph.gar.yml
        graph_yml: Dict[str, Any] = {
            "name": self.book_id,
            "version": "gar/v1",
            "vertices": ["chunk.vertex.yml"],
            "edges": ["chunk_prereq_chunk.edge.yml"],
        }
        (self.graph_dir / "graph.gar.yml").write_text(
            yaml.safe_dump(graph_yml, sort_keys=False),
            encoding="utf-8",
        )

        # 2. Write chunk.vertex.yml
        vertex_yml: Dict[str, Any] = {
            "type": "chunk",
            "chunk_size": self.VERTEX_CHUNK_SIZE,
            "prefix": "vertex/chunk/",
            "version": "gar/v1",
            "property_groups": [
                {
                    "prefix": "metadata/",
                    "file_type": "json_columnar",
                    "properties": [
                        {"name": "chunk_id", "data_type": "string", "is_primary": True},
                        {"name": "chunk_index", "data_type": "int64", "is_primary": False},
                        {"name": "chapter_title", "data_type": "string", "is_primary": False},
                        {"name": "section_title", "data_type": "string", "is_primary": False},
                        {"name": "breadcrumb", "data_type": "string", "is_primary": False},
                        {"name": "chunk_type", "data_type": "string", "is_primary": False},
                        {"name": "token_estimate", "data_type": "int64", "is_primary": False},
                        {"name": "mastery_status", "data_type": "string", "is_primary": False},
                    ],
                }
            ],
        }
        (self.graph_dir / "chunk.vertex.yml").write_text(
            yaml.safe_dump(vertex_yml, sort_keys=False),
            encoding="utf-8",
        )

        # 3. Write chunk_prereq_chunk.edge.yml
        edge_yml: Dict[str, Any] = {
            "src_type": "chunk",
            "edge_type": "prereq",
            "dst_type": "chunk",
            "chunk_size": self.EDGE_CHUNK_SIZE,
            "src_chunk_size": self.VERTEX_CHUNK_SIZE,
            "dst_chunk_size": self.VERTEX_CHUNK_SIZE,
            "directed": True,
            "prefix": "edge/chunk_prereq_chunk/",
            "version": "gar/v1",
            "adj_lists": [
                {
                    "ordered": True,
                    "aligned_by": "src",
                    "prefix": "ordered_by_source/",
                    "file_type": "json_columnar",
                    "property_groups": [
                        {
                            "prefix": "weights/",
                            "file_type": "json_columnar",
                            "properties": [
                                {"name": "similarity_weight", "data_type": "double", "is_primary": False},
                                {"name": "llm_confidence", "data_type": "double", "is_primary": False},
                                {"name": "rationale", "data_type": "string", "is_primary": False},
                                {"name": "bridging_concepts", "data_type": "list<string>", "is_primary": False},
                            ],
                        }
                    ],
                },
                {
                    "ordered": True,
                    "aligned_by": "dst",
                    "prefix": "ordered_by_dest/",
                    "file_type": "json_columnar",
                },
            ],
        }
        (self.graph_dir / "chunk_prereq_chunk.edge.yml").write_text(
            yaml.safe_dump(edge_yml, sort_keys=False),
            encoding="utf-8",
        )

        # 4. Write Vertex Partitions (vertex/chunk/)
        v_dir = self.graph_dir / "vertex" / "chunk"
        v_meta_dir = v_dir / "metadata"
        v_meta_dir.mkdir(parents=True, exist_ok=True)
        (v_dir / "vertex_count").write_text(str(len(vertices)), encoding="utf-8")
        (v_meta_dir / "chunk0.json").write_text(
            json.dumps([v.model_dump(mode="json") for v in vertices], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 5. Write CSR Edge Partitions (ordered_by_source) & CSC Edge Partitions (ordered_by_dest)
        e_base = self.graph_dir / "edge" / "chunk_prereq_chunk"
        csr_dir = e_base / "ordered_by_source"
        csc_dir = e_base / "ordered_by_dest"
        (csr_dir / "adj_list").mkdir(parents=True, exist_ok=True)
        (csr_dir / "offset").mkdir(parents=True, exist_ok=True)
        (csr_dir / "weights").mkdir(parents=True, exist_ok=True)
        (csc_dir / "adj_list").mkdir(parents=True, exist_ok=True)
        (csc_dir / "offset").mkdir(parents=True, exist_ok=True)

        # Build CSR offset array over vertex_index 0..N
        out_counts = [0] * len(vertices)
        in_counts = [0] * len(vertices)
        for e in edges:
            if 0 <= e.source_index < len(vertices):
                out_counts[e.source_index] += 1
            if 0 <= e.target_index < len(vertices):
                in_counts[e.target_index] += 1

        csr_offsets = [0]
        for c in out_counts:
            csr_offsets.append(csr_offsets[-1] + c)

        csc_offsets = [0]
        for c in in_counts:
            csc_offsets.append(csc_offsets[-1] + c)

        (csr_dir / "offset" / "chunk0.json").write_text(json.dumps(csr_offsets), encoding="utf-8")
        (csr_dir / "adj_list" / "chunk0.json").write_text(
            json.dumps([e.model_dump(mode="json") for e in edges], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (csr_dir / "weights" / "chunk0.json").write_text(
            json.dumps(
                [
                    {
                        "source_id": e.source_id,
                        "target_id": e.target_id,
                        "similarity_weight": e.similarity_weight,
                        "llm_confidence": e.llm_confidence,
                        "rationale": e.rationale,
                        "bridging_concepts": e.bridging_concepts,
                    }
                    for e in edges
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        edges_by_dst = sorted(edges, key=lambda e: (e.target_index, e.source_index))
        (csc_dir / "offset" / "chunk0.json").write_text(json.dumps(csc_offsets), encoding="utf-8")
        (csc_dir / "adj_list" / "chunk0.json").write_text(
            json.dumps([e.model_dump(mode="json") for e in edges_by_dst], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return self.graph_dir

    def load(self) -> KnowledgeDiGraph:
        """Load a KnowledgeDiGraph from the Apache GraphAr directory."""
        gar_file = self.graph_dir / "graph.gar.yml"
        if not gar_file.is_file():
            raise FileNotFoundError(f"GraphAr descriptor not found at: {gar_file}")

        v_chunk_path = self.graph_dir / "vertex" / "chunk" / "metadata" / "chunk0.json"
        e_chunk_path = (
            self.graph_dir
            / "edge"
            / "chunk_prereq_chunk"
            / "ordered_by_source"
            / "adj_list"
            / "chunk0.json"
        )

        raw_vertices: List[Dict[str, Any]] = json.loads(v_chunk_path.read_text(encoding="utf-8"))
        raw_edges: List[Dict[str, Any]] = json.loads(e_chunk_path.read_text(encoding="utf-8"))

        graph = KnowledgeDiGraph(book_id=self.book_id)
        for rv in raw_vertices:
            graph.add_vertex(GraphVertex.model_validate(rv))
        for re_item in raw_edges:
            graph.add_edge(PrerequisiteEdge.model_validate(re_item))

        return graph
