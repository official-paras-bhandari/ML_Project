"""
Academic Assignment Generator - Flask Web Application

End-to-end assignment creation tool:
  1. User provides topic or uploads reference docs/PDFs
  2. System researches real sources via academic APIs
  3. User selects citation style (APA, MLA, Harvard, Chicago)
  4. Generates complete, humanized academic assignment with proper citations
  5. Exports as polished DOCX or PDF
"""

import os
import logging
import traceback
from flask import (
    Flask, render_template, request, send_file, jsonify,
    flash, redirect, url_for, session
)
from werkzeug.utils import secure_filename

from modules.researcher import Researcher
from modules.content_generator import ContentGenerator
from modules.document_exporter import DocumentExporter
from modules.citation_engine import CitationEngine

# --- App setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(os.path.dirname(__file__), "outputs")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "txt", "doc", "docx"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Render the main assignment generator form."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    Main generation endpoint.
    Accepts form data with topic, options, and optional file upload.
    Returns a page with download links.
    """
    try:
        # --- Extract form inputs ---
        topic = request.form.get("topic", "").strip()
        citation_style = request.form.get("citation_style", "apa").lower()
        document_type = request.form.get("document_type", "essay").lower()
        word_count = int(request.form.get("word_count", 2000))
        output_format = request.form.get("output_format", "docx").lower()
        num_sources = int(request.form.get("num_sources", 8))

        # Clamp values
        word_count = max(500, min(word_count, 10000))
        num_sources = max(3, min(num_sources, 20))

        if not topic:
            flash("Please enter a topic for your assignment.", "error")
            return redirect(url_for("index"))

        # --- Handle file upload ---
        uploaded_text = ""
        if "reference_file" in request.files:
            file = request.files["reference_file"]
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                logger.info(f"File uploaded: {filename}")

                researcher = Researcher()
                if filename.lower().endswith(".pdf"):
                    uploaded_text = researcher.parse_pdf(filepath)
                elif filename.lower().endswith(".txt"):
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        uploaded_text = f.read()

                # Clean up uploaded file
                try:
                    os.remove(filepath)
                except OSError:
                    pass

        # --- Step 1: Research ---
        logger.info(f"Researching topic: '{topic}' (requesting {num_sources} sources)")
        researcher = Researcher()
        sources = researcher.research_topic(topic, num_sources=num_sources)
        source_count = len(sources)
        logger.info(f"Found {source_count} sources")

        if source_count == 0:
            flash("Could not find academic sources. Please try a different or more specific topic.", "error")
            return redirect(url_for("index"))

        # --- Step 2: Generate content ---
        logger.info(f"Generating {document_type} ({word_count} words, {citation_style} style)")
        generator = ContentGenerator(
            citation_style=citation_style,
            document_type=document_type,
        )
        content = generator.generate(
            topic=topic,
            sources=sources,
            uploaded_text=uploaded_text,
            word_count=word_count,
        )

        # --- Step 3: Export document ---
        exporter = DocumentExporter(output_dir=app.config["OUTPUT_FOLDER"])
        actual_word_count = exporter.get_word_count(content)

        if output_format == "pdf":
            output_path = exporter.export_pdf(content)
        elif output_format == "both":
            docx_path = exporter.export_docx(content)
            pdf_path = exporter.export_pdf(content)
            # For "both", we return the result page with two download links
            return render_template(
                "result.html",
                title=content["title"],
                word_count=actual_word_count,
                source_count=source_count,
                citation_style=citation_style.upper(),
                document_type=document_type.replace("_", " ").title(),
                docx_file=os.path.basename(docx_path),
                pdf_file=os.path.basename(pdf_path),
                sections=content["sections"],
                bibliography=content["bibliography"],
            )
        else:
            output_path = exporter.export_docx(content)

        return render_template(
            "result.html",
            title=content["title"],
            word_count=actual_word_count,
            source_count=source_count,
            citation_style=citation_style.upper(),
            document_type=document_type.replace("_", " ").title(),
            docx_file=os.path.basename(output_path) if output_format == "docx" else None,
            pdf_file=os.path.basename(output_path) if output_format == "pdf" else None,
            sections=content["sections"],
            bibliography=content["bibliography"],
        )

    except Exception as e:
        logger.error(f"Generation error: {traceback.format_exc()}")
        flash(f"An error occurred during generation: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/download/<filename>")
def download(filename):
    """Serve a generated file for download."""
    safe_name = secure_filename(filename)
    filepath = os.path.join(app.config["OUTPUT_FOLDER"], safe_name)
    if not os.path.exists(filepath):
        flash("File not found. It may have been cleaned up.", "error")
        return redirect(url_for("index"))
    return send_file(filepath, as_attachment=True)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    JSON API endpoint for programmatic access.
    Accepts JSON body with: topic, citation_style, document_type, word_count, num_sources
    Returns JSON with generated content and download URLs.
    """
    try:
        data = request.get_json()
        if not data or not data.get("topic"):
            return jsonify({"error": "Topic is required"}), 400

        topic = data["topic"]
        citation_style = data.get("citation_style", "apa")
        document_type = data.get("document_type", "essay")
        word_count = int(data.get("word_count", 2000))
        num_sources = int(data.get("num_sources", 8))

        word_count = max(500, min(word_count, 10000))
        num_sources = max(3, min(num_sources, 20))

        # Research
        researcher = Researcher()
        sources = researcher.research_topic(topic, num_sources=num_sources)

        if not sources:
            return jsonify({"error": "No sources found for this topic"}), 404

        # Generate
        generator = ContentGenerator(citation_style=citation_style, document_type=document_type)
        content = generator.generate(topic=topic, sources=sources, word_count=word_count)

        # Export both formats
        exporter = DocumentExporter(output_dir=app.config["OUTPUT_FOLDER"])
        docx_path = exporter.export_docx(content)
        pdf_path = exporter.export_pdf(content)

        return jsonify({
            "success": True,
            "title": content["title"],
            "word_count": exporter.get_word_count(content),
            "source_count": len(sources),
            "citation_style": citation_style,
            "document_type": document_type,
            "downloads": {
                "docx": url_for("download", filename=os.path.basename(docx_path), _external=True),
                "pdf": url_for("download", filename=os.path.basename(pdf_path), _external=True),
            },
            "sections": [
                {"heading": s["heading"], "preview": s["content"][:200] + "..."}
                for s in content["sections"]
            ],
            "bibliography": content["bibliography"],
            "sources": researcher.get_source_summaries(),
        })

    except Exception as e:
        logger.error(f"API error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
