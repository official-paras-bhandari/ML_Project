"""Quick integration test for the assignment generator pipeline."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from modules.researcher import Researcher
from modules.content_generator import ContentGenerator
from modules.document_exporter import DocumentExporter
from modules.citation_engine import CitationEngine, Source


def test_citation_engine():
    """Test all 4 citation styles."""
    source = Source(
        title="The Impact of AI on Modern Healthcare",
        authors=["John Smith", "Jane Doe"],
        year="2023",
        journal="Journal of Medical Informatics",
        volume="45",
        issue="2",
        pages="112-128",
        doi="10.1234/jmi.2023.001",
        source_type="journal",
    )

    for style in ["apa", "mla", "harvard", "chicago"]:
        engine = CitationEngine(style)
        inline = engine.format_inline(source)
        ref = engine.format_reference(source)
        print(f"\n[{style.upper()}]")
        print(f"  Inline: {inline}")
        print(f"  Reference: {ref}")
        assert inline, f"Empty inline citation for {style}"
        assert ref, f"Empty reference for {style}"

    print("\n[PASS] Citation engine works for all styles.")


def test_researcher():
    """Test the research module."""
    r = Researcher()
    sources = r.research_topic("artificial intelligence healthcare", num_sources=6)
    print(f"\nResearch found {len(sources)} sources:")
    for s in sources[:5]:
        print(f"  - {s.title} ({s.year}) [{s.source_type}]")
    assert len(sources) > 0, "No sources found"
    print("\n[PASS] Researcher found real sources.")
    return sources


def test_content_generator(sources):
    """Test content generation."""
    gen = ContentGenerator(citation_style="apa", document_type="essay")
    content = gen.generate(
        topic="Artificial Intelligence in Healthcare",
        sources=sources,
        word_count=1500,
    )
    print(f"\nGenerated: {content['title']}")
    print(f"Sections: {len(content['sections'])}")
    total_words = sum(len(s['content'].split()) for s in content['sections'])
    print(f"Total words: {total_words}")
    print(f"Bibliography entries: {len(content['bibliography'])}")
    assert content['title'], "No title"
    assert len(content['sections']) > 0, "No sections"
    assert total_words > 500, f"Too few words: {total_words}"
    print("\n[PASS] Content generator works.")
    return content


def test_document_export(content):
    """Test DOCX and PDF export."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    exporter = DocumentExporter(output_dir=output_dir)

    docx_path = exporter.export_docx(content, filename="test_output.docx")
    assert os.path.exists(docx_path), "DOCX not created"
    print(f"\nDOCX created: {docx_path} ({os.path.getsize(docx_path)} bytes)")

    pdf_path = exporter.export_pdf(content, filename="test_output.pdf")
    assert os.path.exists(pdf_path), "PDF not created"
    print(f"PDF created: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

    print("\n[PASS] Document export works (DOCX + PDF).")


if __name__ == "__main__":
    print("=" * 60)
    print("ACADEMIC ASSIGNMENT GENERATOR - Integration Test")
    print("=" * 60)

    test_citation_engine()
    sources = test_researcher()
    content = test_content_generator(sources)
    test_document_export(content)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
