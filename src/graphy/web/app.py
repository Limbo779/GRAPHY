"""Flask + Waitress Local Web UI & REST API for GRAPHY and GRAPHY Companion."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from graphy.companion.pipeline import ingest_problem_set
from graphy.core.config import GraphyConfig, load_config
from graphy.core.hardware import probe_hardware
from graphy.embeddings.engine import EmbeddingMode
from graphy.graph.prerequisite_builder import build_prerequisite_graph
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.parser.pipeline import ingest_textbook
from graphy.scoring.mastery_updater import load_saved_scores, score_book_and_problems
from graphy.tutor.session_manager import StudySessionManager


def create_web_app(
    config: Optional[GraphyConfig] = None,
    model_manager: Optional[ModelLifecycleManager] = None,
    embedding_mode: EmbeddingMode = "hybrid",
    base_dir: Optional[Path] = None,
) -> Flask:
    """Create and configure the lightweight local Flask web application."""
    cfg = config or load_config()
    root = base_dir or Path(".")
    cfg.ensure_Runtime_dirs(base_dir=root)
    data_root = root / cfg.system.data_dir

    template_dir = Path(__file__).resolve().parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/health")
    def api_health() -> Response:
        hw = probe_hardware(
            dri_node=cfg.hardware.gpu.dri_render_node,
            max_container_cap_gib=cfg.hardware.max_container_ram_gib,
        )
        return jsonify(
            {
                "status": "ok",
                "project": cfg.system.project_name,
                "in_container": hw.in_container,
                "cpu_model": hw.cpu.model_name,
                "logical_cpus": hw.cpu.logical_cpus,
                "safe_ram_gib": hw.memory.safe_working_set_gib,
                "iris_xe": hw.gpu.is_intel_iris_xe,
                "vulkan": hw.gpu.vulkan_available,
            }
        )

    @app.get("/api/books")
    def api_list_books() -> Response:
        scores_dir = data_root / "scores"
        vectors_dir = data_root / "vectors"
        book_ids = set()
        if scores_dir.is_dir():
            for d in scores_dir.iterdir():
                if d.is_dir():
                    book_ids.add(d.name)
        if vectors_dir.is_dir():
            for d in vectors_dir.iterdir():
                if d.is_dir():
                    book_ids.add(d.name)
        return jsonify({"books": sorted(book_ids)})

    @app.post("/api/ingest-book")
    def api_ingest_book() -> tuple[Response, int]:
        payload = request.get_json(silent=True) or {}
        file_path = payload.get("file_path")
        book_id = payload.get("book_id")
        if not file_path or not book_id:
            return jsonify({"error": "Both 'file_path' and 'book_id' are required."}), 400

        summary = ingest_textbook(
            file_path=file_path,
            book_id=book_id,
            config=cfg,
            embedding_mode=embedding_mode,
            model_manager=model_manager,
            base_dir=root,
        )
        return jsonify(summary.model_dump(mode="json")), 200

    @app.post("/api/ingest-problems")
    def api_ingest_problems() -> tuple[Response, int]:
        payload = request.get_json(silent=True) or {}
        file_path = payload.get("file_path")
        book_id = payload.get("book_id")
        source_id = payload.get("source_id")
        if not file_path or not book_id:
            return jsonify({"error": "Both 'file_path' and 'book_id' are required."}), 400

        summary = ingest_problem_set(
            file_path=file_path,
            book_id=book_id,
            source_id=source_id,
            config=cfg,
            base_dir=root,
        )
        return jsonify(summary.model_dump(mode="json")), 200

    @app.post("/api/build-graph/<book_id>")
    def api_build_graph(book_id: str) -> tuple[Response, int]:
        graph_sum = build_prerequisite_graph(
            book_id=book_id,
            config=cfg,
            model_manager=model_manager,
            base_dir=root,
        )
        score_sum = score_book_and_problems(
            book_id=book_id,
            config=cfg,
            embedding_mode=embedding_mode,
            model_manager=model_manager,
            base_dir=root,
        )
        return (
            jsonify(
                {
                    "graph": graph_sum.model_dump(mode="json"),
                    "scoring": score_sum.model_dump(mode="json"),
                }
            ),
            200,
        )

    @app.get("/api/books/<book_id>/scores")
    def api_book_scores(book_id: str) -> tuple[Response, int]:
        try:
            cs, pcs, pfs = load_saved_scores(book_id=book_id, base_dir=root, config=cfg)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

        return (
            jsonify(
                {
                    "book_id": book_id,
                    "chunk_scores": [c.model_dump(mode="json") for c in cs],
                    "problem_chunk_scores": [p.model_dump(mode="json") for p in pcs],
                    "problem_foundation_scores": [p.model_dump(mode="json") for p in pfs],
                }
            ),
            200,
        )

    @app.post("/api/books/<book_id>/study/start")
    def api_study_start(book_id: str) -> tuple[Response, int]:
        tutor = StudySessionManager(
            book_id=book_id,
            config=cfg,
            model_manager=model_manager,
            embedding_mode=embedding_mode,
            base_dir=root,
        )
        session = tutor.start_next_batch()
        if session is None:
            return jsonify({"error": "All problems for this book are already solved!"}), 400
        briefing = session.to_learner_briefing()
        return jsonify(briefing.model_dump(mode="json")), 200

    @app.post("/api/books/<book_id>/study/<session_id>/ocr-preview")
    def api_study_ocr_preview(book_id: str, session_id: str) -> tuple[Response, int]:
        tutor = StudySessionManager(
            book_id=book_id,
            config=cfg,
            model_manager=model_manager,
            embedding_mode=embedding_mode,
            base_dir=root,
        )

        # Support both multipart file upload and JSON {"problem_id": ..., "image_path": ...}
        if "image" in request.files:
            problem_id = request.form.get("problem_id", "")
            img_file = request.files["image"]
            uploads_dir = data_root / "uploads" / book_id
            uploads_dir.mkdir(parents=True, exist_ok=True)
            safe_name = secure_filename(img_file.filename or f"{problem_id}.png")
            saved_img_path = uploads_dir / f"{session_id}_{problem_id}_{safe_name}"
            img_file.save(str(saved_img_path))
        else:
            body = request.get_json(silent=True) or {}
            problem_id = body.get("problem_id", "")
            saved_img_path = Path(body.get("image_path", ""))

        if not problem_id or not saved_img_path.is_file():
            return jsonify({"error": "Valid 'problem_id' and image file are required."}), 400

        preview = tutor.preview_handwritten_solution(
            session_id=session_id,
            problem_id=problem_id,
            image_path=saved_img_path,
            use_vlm=True,
        )
        return jsonify(preview.model_dump(mode="json")), 200

    @app.post("/api/books/<book_id>/study/<session_id>/ocr-confirm")
    def api_study_ocr_confirm(book_id: str, session_id: str) -> tuple[Response, int]:
        body = request.get_json(silent=True) or {}
        problem_id = body.get("problem_id", "")
        edited_latex_md = body.get("edited_latex_md")
        if not problem_id:
            return jsonify({"error": "'problem_id' is required."}), 400

        tutor = StudySessionManager(
            book_id=book_id,
            config=cfg,
            model_manager=model_manager,
            embedding_mode=embedding_mode,
            base_dir=root,
        )
        preview = tutor.confirm_handwritten_solution(
            session_id=session_id,
            problem_id=problem_id,
            edited_latex_md=edited_latex_md,
        )
        return jsonify(preview.model_dump(mode="json")), 200

    @app.post("/api/books/<book_id>/study/<session_id>/evaluate")
    def api_study_evaluate(book_id: str, session_id: str) -> tuple[Response, int]:
        body = request.get_json(silent=True) or {}
        solutions = body.get("solutions", {})

        tutor = StudySessionManager(
            book_id=book_id,
            config=cfg,
            model_manager=model_manager,
            embedding_mode=embedding_mode,
            base_dir=root,
        )
        if solutions:
            tutor.submit_text_solutions(
                session_id=session_id,
                solutions_by_problem_id=solutions,
            )

        try:
            report = tutor.evaluate_batch_solutions(session_id=session_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(report.model_dump(mode="json")), 200

    return app


def run_web_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    config: Optional[GraphyConfig] = None,
) -> None:
    """Serve the local Web UI via Waitress WSGI server."""
    import waitress

    cfg = config or load_config()
    app = create_web_app(config=cfg)
    print(f"GRAPHY Local Web UI listening on http://{host}:{port}")
    waitress.serve(app, host=host, port=port, threads=4)
