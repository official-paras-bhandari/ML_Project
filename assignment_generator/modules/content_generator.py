"""
Content Generator - Produces humanized academic writing with proper citations.

Uses a template-based approach with variation patterns to produce natural-sounding
academic prose. Integrates real sources via inline citations.
"""

import random
import re
from .citation_engine import CitationEngine, Source


# --- Sentence variation templates for natural academic writing ---

INTRO_TEMPLATES = [
    "The topic of {topic} has attracted considerable attention in recent academic discourse.",
    "In contemporary scholarship, {topic} represents a subject of growing significance.",
    "{topic} has emerged as a critical area of inquiry across multiple disciplines.",
    "Understanding {topic} remains essential for both researchers and practitioners.",
    "Recent developments in {topic} have prompted renewed scholarly interest.",
    "The study of {topic} continues to evolve as new evidence and perspectives emerge.",
    "Academic interest in {topic} has intensified in recent years, driven by emerging challenges.",
    "Few subjects in modern academia have generated as much discussion as {topic}.",
]

TRANSITION_PHRASES = [
    "Furthermore, ", "Moreover, ", "In addition, ", "Building on this, ",
    "Equally important, ", "It is also worth noting that ",
    "Along similar lines, ", "Extending this perspective, ",
    "A related consideration involves ", "This point is further supported by the fact that ",
]

EVIDENCE_TEMPLATES = [
    "According to {citation}, {claim}.",
    "Research by {author} suggests that {claim} {citation}.",
    "As {author} argues, {claim} {citation}.",
    "{claim}, as demonstrated in recent research {citation}.",
    "Evidence from the literature indicates that {claim} {citation}.",
    "It has been observed that {claim} {citation}.",
    "Scholars have noted that {claim} {citation}.",
    "The work of {author} highlights that {claim} {citation}.",
]

ANALYSIS_TEMPLATES = [
    "This suggests that {analysis}.",
    "Such findings imply that {analysis}.",
    "These observations point toward the conclusion that {analysis}.",
    "This evidence supports the notion that {analysis}.",
    "The significance of this lies in the fact that {analysis}.",
    "This is particularly relevant because {analysis}.",
    "What emerges from this analysis is that {analysis}.",
]

CONCLUSION_TEMPLATES = [
    "In summary, this {doc_type} has examined {topic} through multiple lenses.",
    "To conclude, the exploration of {topic} reveals several important insights.",
    "This {doc_type} has sought to provide a comprehensive overview of {topic}.",
    "In closing, the analysis presented here underscores the complexity of {topic}.",
    "Drawing together the threads of this discussion, it becomes clear that {topic} merits continued attention.",
]

SECTION_HEADERS_ESSAY = [
    "Introduction",
    "Literature Review",
    "Discussion",
    "Analysis",
    "Conclusion",
]

SECTION_HEADERS_REPORT = [
    "Introduction",
    "Background",
    "Methodology Overview",
    "Findings and Analysis",
    "Discussion",
    "Conclusion and Recommendations",
]

SECTION_HEADERS_PAPER = [
    "Abstract",
    "Introduction",
    "Literature Review",
    "Theoretical Framework",
    "Discussion",
    "Conclusion",
]

HEDGING_PHRASES = [
    "It appears that", "Evidence suggests that", "It is likely that",
    "There is reason to believe that", "It could be argued that",
    "One might consider that", "It seems reasonable to conclude that",
    "Based on available evidence,", "From this perspective,",
]

TOPIC_CLAIMS = {
    "_default": [
        "significant progress has been made in understanding key aspects of this field",
        "multiple factors contribute to the complexity of this subject",
        "ongoing research continues to reveal new dimensions of this topic",
        "practical applications have emerged from theoretical foundations",
        "interdisciplinary approaches have enriched our understanding",
        "challenges remain in fully addressing all aspects of this issue",
        "contemporary perspectives offer valuable insights into this domain",
        "the relationship between theory and practice remains a central concern",
        "further investigation is warranted to address remaining gaps in knowledge",
        "existing frameworks provide a useful but incomplete picture",
        "the implications extend beyond the immediate field of study",
        "there is growing consensus on several fundamental aspects",
        "methodological advances have enabled more nuanced analyses",
        "the historical context provides important background for current debates",
        "stakeholder perspectives vary considerably on this matter",
    ]
}


class ContentGenerator:
    """Generates humanized academic content with integrated citations."""

    DOCUMENT_TYPES = ["essay", "report", "research_paper"]

    def __init__(self, citation_style="apa", document_type="essay"):
        self.citation_engine = CitationEngine(citation_style)
        self.document_type = document_type
        self._used_templates = set()

    def generate(self, topic, sources, uploaded_text="", word_count=2000):
        """
        Generate a complete academic document.

        Returns a dict with:
          - title: str
          - sections: list of {heading: str, content: str}
          - bibliography: list of str
        """
        headers = self._get_section_headers()
        title = self._generate_title(topic)
        sections = []

        # Distribute sources across sections
        body_sections = [h for h in headers if h.lower() not in ("abstract",)]
        words_per_section = word_count // len(body_sections)

        source_groups = self._distribute_sources(sources, len(body_sections))

        for i, heading in enumerate(body_sections):
            section_sources = source_groups[i] if i < len(source_groups) else []
            content = self._generate_section(
                topic=topic,
                heading=heading,
                sources=section_sources,
                all_sources=sources,
                uploaded_text=uploaded_text,
                target_words=words_per_section,
                is_first=(i == 0),
                is_last=(i == len(body_sections) - 1),
            )
            sections.append({"heading": heading, "content": content})

        # Abstract (for research papers)
        if "Abstract" in headers:
            abstract = self._generate_abstract(topic, sections)
            sections.insert(0, {"heading": "Abstract", "content": abstract})

        bibliography = self.citation_engine.format_bibliography(sources)

        return {
            "title": title,
            "sections": sections,
            "bibliography": bibliography,
        }

    def _get_section_headers(self):
        if self.document_type == "report":
            return SECTION_HEADERS_REPORT
        elif self.document_type == "research_paper":
            return SECTION_HEADERS_PAPER
        return SECTION_HEADERS_ESSAY

    def _generate_title(self, topic):
        prefixes = [
            "An Examination of", "Exploring", "Understanding",
            "A Critical Analysis of", "Perspectives on",
            "Investigating", "The Role of", "Rethinking",
        ]
        return f"{random.choice(prefixes)} {topic}"

    def _distribute_sources(self, sources, num_sections):
        """Distribute sources roughly evenly across sections."""
        groups = [[] for _ in range(num_sections)]
        for i, source in enumerate(sources):
            groups[i % num_sections].append(source)
        return groups

    def _generate_section(self, topic, heading, sources, all_sources,
                          uploaded_text, target_words, is_first, is_last):
        """Generate content for a single section."""
        paragraphs = []
        heading_lower = heading.lower()

        if is_first or "introduction" in heading_lower:
            paragraphs.append(self._write_introduction(topic, all_sources))
        elif is_last or "conclusion" in heading_lower:
            paragraphs.append(self._write_conclusion(topic))
        elif "literature" in heading_lower or "background" in heading_lower:
            paragraphs.extend(self._write_literature_review(topic, sources))
        elif "method" in heading_lower:
            paragraphs.append(self._write_methodology(topic))
        elif "discussion" in heading_lower or "analysis" in heading_lower or "findings" in heading_lower:
            paragraphs.extend(self._write_discussion(topic, sources))
        elif "recommendation" in heading_lower:
            paragraphs.append(self._write_recommendations(topic))
        elif "framework" in heading_lower:
            paragraphs.extend(self._write_framework(topic, sources))
        else:
            paragraphs.extend(self._write_generic_section(topic, sources))

        # If uploaded text exists, weave in key points
        if uploaded_text and ("discussion" in heading_lower or "analysis" in heading_lower):
            from .researcher import Researcher
            r = Researcher()
            key_points = r.extract_key_points(uploaded_text, max_points=3)
            if key_points:
                extra = self._integrate_uploaded_content(key_points)
                paragraphs.append(extra)

        # Pad or trim to approximate target word count
        content = "\n\n".join(paragraphs)
        current_words = len(content.split())

        while current_words < target_words * 0.7:
            extra = self._generate_filler_paragraph(topic, sources)
            content += "\n\n" + extra
            current_words = len(content.split())

        return content

    def _pick_template(self, templates):
        """Pick a template, avoiding recent repeats."""
        available = [t for t in templates if t not in self._used_templates]
        if not available:
            self._used_templates.clear()
            available = templates
        choice = random.choice(available)
        self._used_templates.add(choice)
        return choice

    def _get_claim(self, topic):
        """Get a relevant claim for the topic."""
        claims = TOPIC_CLAIMS.get("_default", [])
        return random.choice(claims)

    def _write_introduction(self, topic, sources):
        """Write an introduction paragraph."""
        intro = self._pick_template(INTRO_TEMPLATES).format(topic=topic)

        # Add a sentence referencing scope
        scope = (
            f" This {self.document_type.replace('_', ' ')} aims to explore "
            f"the key dimensions of {topic}, drawing upon recent scholarly work "
            f"and empirical evidence."
        )

        # Add a sentence about what the document covers
        structure = (
            f" The discussion begins with a review of relevant literature, "
            f"followed by a detailed analysis, before concluding with "
            f"implications and potential directions for future inquiry."
        )

        # Cite an early source if available
        cite_sentence = ""
        if sources:
            s = sources[0]
            cite = self.citation_engine.format_inline(s)
            claim = self._get_claim(topic)
            cite_sentence = f" As noted in prior research, {claim} {cite}."

        return intro + scope + cite_sentence + structure

    def _write_literature_review(self, topic, sources):
        """Write literature review paragraphs."""
        paragraphs = []

        opening = (
            f"A review of the existing literature on {topic} reveals "
            f"several important themes and findings that inform the present discussion."
        )
        paragraphs.append(opening)

        for source in sources:
            claim = self._get_claim(topic)
            author_name = source.authors[0].split()[-1] if source.authors else "researchers"
            citation = self.citation_engine.format_inline(source)

            template = self._pick_template(EVIDENCE_TEMPLATES)
            sentence = template.format(
                citation=citation,
                author=author_name,
                claim=claim,
            )

            analysis_template = self._pick_template(ANALYSIS_TEMPLATES)
            analysis = analysis_template.format(
                analysis=self._get_claim(topic)
            )

            transition = random.choice(TRANSITION_PHRASES) if paragraphs else ""
            paragraph = f"{transition}{sentence} {analysis}"
            paragraphs.append(paragraph)

        return paragraphs

    def _write_discussion(self, topic, sources):
        """Write discussion/analysis paragraphs."""
        paragraphs = []

        opening = (
            f"The analysis of {topic} presented here draws on the evidence "
            f"reviewed in the preceding sections. Several key themes emerge "
            f"from this synthesis."
        )
        paragraphs.append(opening)

        for i, source in enumerate(sources):
            hedge = random.choice(HEDGING_PHRASES)
            claim = self._get_claim(topic)
            citation = self.citation_engine.format_inline(source)

            paragraph = f"{hedge} {claim} {citation}. "
            paragraph += self._pick_template(ANALYSIS_TEMPLATES).format(
                analysis=self._get_claim(topic)
            )

            if i < len(sources) - 1:
                paragraph += f" {random.choice(TRANSITION_PHRASES).strip().lower()}"
                paragraph += f"{self._get_claim(topic)}."

            paragraphs.append(paragraph)

        return paragraphs

    def _write_methodology(self, topic):
        """Write a methodology overview paragraph."""
        return (
            f"This {self.document_type.replace('_', ' ')} employs a qualitative "
            f"approach to examining {topic}, relying primarily on a systematic "
            f"review of existing literature and secondary data analysis. "
            f"Sources were selected based on their relevance, recency, and "
            f"scholarly credibility. The analysis follows a thematic structure, "
            f"identifying and synthesizing key patterns across the reviewed materials. "
            f"This methodological approach enables a comprehensive yet focused "
            f"examination of the subject matter."
        )

    def _write_framework(self, topic, sources):
        """Write theoretical framework paragraphs."""
        paragraphs = []
        opening = (
            f"The theoretical underpinnings of {topic} draw from several "
            f"established frameworks in the field. Understanding these foundations "
            f"is essential for a rigorous analysis."
        )
        paragraphs.append(opening)

        for source in sources[:3]:
            citation = self.citation_engine.format_inline(source)
            claim = self._get_claim(topic)
            author_name = source.authors[0].split()[-1] if source.authors else "scholars"
            paragraph = (
                f"The work of {author_name} provides a useful lens through which "
                f"to examine this topic. {claim.capitalize()} {citation}. "
                f"This perspective contributes to a more nuanced understanding "
                f"of the underlying dynamics at play."
            )
            paragraphs.append(paragraph)

        return paragraphs

    def _write_recommendations(self, topic):
        """Write recommendations paragraph."""
        return (
            f"Based on the analysis presented in this {self.document_type.replace('_', ' ')}, "
            f"several recommendations can be offered. First, further research is needed "
            f"to address the identified gaps in the current understanding of {topic}. "
            f"Second, practitioners should consider the evidence-based insights discussed "
            f"here when developing strategies and policies. Third, interdisciplinary "
            f"collaboration may yield valuable new perspectives on this subject. "
            f"Finally, stakeholders are encouraged to engage with the ongoing scholarly "
            f"discourse to ensure that decisions are informed by the most current evidence."
        )

    def _write_conclusion(self, topic):
        """Write a conclusion paragraph."""
        template = self._pick_template(CONCLUSION_TEMPLATES)
        doc_type = self.document_type.replace("_", " ")
        conclusion = template.format(topic=topic, doc_type=doc_type)

        closing = (
            f" The evidence reviewed and analyzed throughout this work "
            f"demonstrates that {topic} is a multifaceted subject with "
            f"significant implications. While this {doc_type} has addressed "
            f"several core aspects, the breadth and depth of the topic "
            f"suggest ample opportunity for future scholarship. It is hoped "
            f"that the insights presented here will contribute meaningfully "
            f"to the ongoing conversation surrounding {topic}."
        )
        return conclusion + closing

    def _write_generic_section(self, topic, sources):
        """Write a generic section when heading doesn't match known types."""
        paragraphs = []
        for source in sources:
            claim = self._get_claim(topic)
            citation = self.citation_engine.format_inline(source)
            template = self._pick_template(EVIDENCE_TEMPLATES)
            author_name = source.authors[0].split()[-1] if source.authors else "researchers"
            sentence = template.format(citation=citation, author=author_name, claim=claim)
            analysis = self._pick_template(ANALYSIS_TEMPLATES).format(analysis=self._get_claim(topic))
            paragraphs.append(f"{sentence} {analysis}")
        if not paragraphs:
            paragraphs.append(
                f"An exploration of {topic} at this stage reveals important "
                f"considerations that warrant further discussion. "
                + self._pick_template(ANALYSIS_TEMPLATES).format(
                    analysis=self._get_claim(topic)
                )
            )
        return paragraphs

    def _generate_filler_paragraph(self, topic, sources):
        """Generate an additional paragraph to reach target word count."""
        hedge = random.choice(HEDGING_PHRASES)
        claim = self._get_claim(topic)
        analysis = self._pick_template(ANALYSIS_TEMPLATES).format(
            analysis=self._get_claim(topic)
        )

        cite_part = ""
        if sources:
            s = random.choice(sources)
            cite_part = f" {self.citation_engine.format_inline(s)}"

        return f"{hedge} {claim}{cite_part}. {analysis}"

    def _generate_abstract(self, topic, sections):
        """Generate an abstract for research papers."""
        return (
            f"This paper examines {topic} through a comprehensive review of "
            f"existing literature and secondary analysis. The study identifies "
            f"key themes, evaluates current evidence, and synthesizes findings "
            f"from multiple scholarly sources. Results indicate that {topic} "
            f"involves complex, interrelated factors that require nuanced "
            f"understanding. The paper concludes with recommendations for "
            f"future research and practical implications for stakeholders. "
            f"Keywords: {topic.lower()}, literature review, analysis, research."
        )

    def _integrate_uploaded_content(self, key_points):
        """Weave uploaded document key points into a paragraph."""
        intro = "Drawing from the provided reference materials, several additional points merit discussion. "
        sentences = []
        for point in key_points[:3]:
            # Clean and rephrase slightly
            clean = point.strip().rstrip(".")
            sentences.append(f"Notably, {clean.lower()}" if not clean[0].isupper() else f"As noted, {clean}")
        return intro + ". ".join(sentences) + "."

    def _generate_abstract(self, topic, sections):
        """Generate an abstract summarizing the paper."""
        return (
            f"This paper presents a comprehensive examination of {topic}. "
            f"Through a systematic review of relevant literature and careful analysis, "
            f"this study explores the key dimensions, challenges, and implications "
            f"associated with {topic}. The findings draw upon multiple academic "
            f"sources to provide a well-rounded perspective. The paper concludes "
            f"with a discussion of implications and directions for future research. "
            f"Keywords: {topic.lower()}, literature review, academic analysis."
        )
