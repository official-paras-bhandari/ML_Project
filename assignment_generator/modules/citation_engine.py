"""
Citation Engine - Formats references in APA, MLA, Harvard, and Chicago styles.
"""

from datetime import datetime


class Source:
    """Represents a research source with metadata."""

    def __init__(self, title="", authors=None, year="", url="", journal="",
                 volume="", issue="", pages="", publisher="", doi="",
                 accessed_date=None, source_type="web"):
        self.title = title
        self.authors = authors or []
        self.year = str(year) if year else ""
        self.url = url
        self.journal = journal
        self.volume = volume
        self.issue = issue
        self.pages = pages
        self.publisher = publisher
        self.doi = doi
        self.accessed_date = accessed_date or datetime.now().strftime("%B %d, %Y")
        self.source_type = source_type  # "web", "journal", "book"

    def to_dict(self):
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "url": self.url,
            "journal": self.journal,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "publisher": self.publisher,
            "doi": self.doi,
            "accessed_date": self.accessed_date,
            "source_type": self.source_type,
        }


def _format_author_apa(author):
    """Format a single author name for APA: Last, F. M."""
    parts = author.strip().split()
    if len(parts) == 0:
        return ""
    if len(parts) == 1:
        return parts[0]
    last = parts[-1]
    initials = " ".join(f"{p[0]}." for p in parts[:-1])
    return f"{last}, {initials}"


def _format_authors_apa(authors):
    if not authors:
        return ""
    if len(authors) == 1:
        return _format_author_apa(authors[0])
    if len(authors) == 2:
        return f"{_format_author_apa(authors[0])}, & {_format_author_apa(authors[1])}"
    if len(authors) <= 20:
        formatted = ", ".join(_format_author_apa(a) for a in authors[:-1])
        return f"{formatted}, & {_format_author_apa(authors[-1])}"
    formatted = ", ".join(_format_author_apa(a) for a in authors[:19])
    return f"{formatted}, ... {_format_author_apa(authors[-1])}"


def _format_author_mla(author):
    """Format a single author name for MLA: Last, First Middle."""
    parts = author.strip().split()
    if len(parts) <= 1:
        return author.strip()
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def _format_authors_mla(authors):
    if not authors:
        return ""
    if len(authors) == 1:
        return _format_author_mla(authors[0])
    if len(authors) == 2:
        return f"{_format_author_mla(authors[0])}, and {authors[1].strip()}"
    return f"{_format_author_mla(authors[0])}, et al."


def _format_authors_harvard(authors):
    if not authors:
        return ""
    if len(authors) == 1:
        return _format_author_apa(authors[0])
    if len(authors) == 2:
        return f"{_format_author_apa(authors[0])} and {_format_author_apa(authors[1])}"
    if len(authors) <= 3:
        formatted = ", ".join(_format_author_apa(a) for a in authors[:-1])
        return f"{formatted} and {_format_author_apa(authors[-1])}"
    return f"{_format_author_apa(authors[0])} et al."


def _format_author_chicago(author):
    parts = author.strip().split()
    if len(parts) <= 1:
        return author.strip()
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def _format_authors_chicago(authors):
    if not authors:
        return ""
    if len(authors) == 1:
        return _format_author_chicago(authors[0])
    if len(authors) <= 3:
        formatted = [_format_author_chicago(authors[0])]
        formatted.extend(a.strip() for a in authors[1:])
        return ", and ".join([", ".join(formatted[:-1]), formatted[-1]]) if len(formatted) > 2 else " and ".join(formatted)
    return f"{_format_author_chicago(authors[0])} et al."


class CitationEngine:
    """Generates formatted citations and in-text references."""

    SUPPORTED_STYLES = ["apa", "mla", "harvard", "chicago"]

    def __init__(self, style="apa"):
        style = style.lower()
        if style not in self.SUPPORTED_STYLES:
            raise ValueError(f"Unsupported style: {style}. Choose from {self.SUPPORTED_STYLES}")
        self.style = style

    def format_inline(self, source, page=None):
        """Generate an in-text citation string."""
        if self.style == "apa":
            return self._inline_apa(source, page)
        elif self.style == "mla":
            return self._inline_mla(source, page)
        elif self.style == "harvard":
            return self._inline_harvard(source, page)
        elif self.style == "chicago":
            return self._inline_chicago(source, page)

    def format_reference(self, source):
        """Generate a full reference list entry."""
        if self.style == "apa":
            return self._ref_apa(source)
        elif self.style == "mla":
            return self._ref_mla(source)
        elif self.style == "harvard":
            return self._ref_harvard(source)
        elif self.style == "chicago":
            return self._ref_chicago(source)

    def format_bibliography(self, sources):
        """Generate a sorted bibliography/reference list."""
        refs = [self.format_reference(s) for s in sources]
        refs.sort(key=lambda x: x.lower())
        return refs

    # --- APA 7th Edition ---
    def _inline_apa(self, source, page=None):
        if source.authors:
            last = source.authors[0].strip().split()[-1]
            if len(source.authors) == 2:
                last2 = source.authors[1].strip().split()[-1]
                name_part = f"{last} & {last2}"
            elif len(source.authors) > 2:
                name_part = f"{last} et al."
            else:
                name_part = last
        else:
            name_part = f'"{source.title[:30]}..."' if len(source.title) > 30 else f'"{source.title}"'
        year_part = source.year if source.year else "n.d."
        if page:
            return f"({name_part}, {year_part}, p. {page})"
        return f"({name_part}, {year_part})"

    def _ref_apa(self, source):
        authors = _format_authors_apa(source.authors) if source.authors else ""
        year = f"({source.year})" if source.year else "(n.d.)"

        if source.source_type == "journal":
            title = source.title
            journal = f"*{source.journal}*" if source.journal else ""
            vol_issue = ""
            if source.volume:
                vol_issue = f", *{source.volume}*"
                if source.issue:
                    vol_issue += f"({source.issue})"
            pages = f", {source.pages}" if source.pages else ""
            doi_url = f" https://doi.org/{source.doi}" if source.doi else (f" {source.url}" if source.url else "")
            return f"{authors} {year}. {title}. {journal}{vol_issue}{pages}.{doi_url}".strip()

        elif source.source_type == "book":
            title = f"*{source.title}*"
            publisher = f" {source.publisher}." if source.publisher else ""
            doi_url = f" https://doi.org/{source.doi}" if source.doi else (f" {source.url}" if source.url else "")
            return f"{authors} {year}. {title}.{publisher}{doi_url}".strip()

        else:  # web
            title = f"*{source.title}*"
            site = f" {source.publisher}." if source.publisher else ""
            url_part = f" {source.url}" if source.url else ""
            return f"{authors} {year}. {title}.{site}{url_part}".strip()

    # --- MLA 9th Edition ---
    def _inline_mla(self, source, page=None):
        if source.authors:
            last = source.authors[0].strip().split()[-1]
            if len(source.authors) > 2:
                name_part = f"{last} et al."
            elif len(source.authors) == 2:
                last2 = source.authors[1].strip().split()[-1]
                name_part = f"{last} and {last2}"
            else:
                name_part = last
        else:
            title_short = source.title[:30] + "..." if len(source.title) > 30 else source.title
            name_part = f'"{title_short}"'
        if page:
            return f"({name_part} {page})"
        return f"({name_part})"

    def _ref_mla(self, source):
        authors = _format_authors_mla(source.authors) if source.authors else ""

        if source.source_type == "journal":
            title = f'"{source.title}."'
            journal = f"*{source.journal}*," if source.journal else ""
            vol = f" vol. {source.volume}," if source.volume else ""
            issue = f" no. {source.issue}," if source.issue else ""
            year = f" {source.year}," if source.year else ""
            pages = f" pp. {source.pages}." if source.pages else "."
            doi_url = f" {source.doi}" if source.doi else (f" {source.url}" if source.url else "")
            return f"{authors}. {title} {journal}{vol}{issue}{year}{pages}{doi_url}".strip()

        elif source.source_type == "book":
            title = f"*{source.title}*."
            publisher = f" {source.publisher}," if source.publisher else ""
            year = f" {source.year}." if source.year else "."
            return f"{authors}. {title}{publisher}{year}".strip()

        else:
            title = f'"{source.title}."'
            site = f" *{source.publisher}*," if source.publisher else ""
            year = f" {source.year}," if source.year else ""
            url = f" {source.url}." if source.url else "."
            accessed = f" Accessed {source.accessed_date}." if source.url else ""
            return f"{authors}. {title}{site}{year}{url}{accessed}".strip()

    # --- Harvard ---
    def _inline_harvard(self, source, page=None):
        if source.authors:
            last = source.authors[0].strip().split()[-1]
            if len(source.authors) > 3:
                name_part = f"{last} et al."
            elif len(source.authors) == 2:
                last2 = source.authors[1].strip().split()[-1]
                name_part = f"{last} and {last2}"
            elif len(source.authors) == 3:
                lasts = [a.strip().split()[-1] for a in source.authors]
                name_part = f"{lasts[0]}, {lasts[1]} and {lasts[2]}"
            else:
                name_part = last
        else:
            name_part = source.title[:30] + "..." if len(source.title) > 30 else source.title
        year = source.year if source.year else "n.d."
        if page:
            return f"({name_part}, {year}, p. {page})"
        return f"({name_part}, {year})"

    def _ref_harvard(self, source):
        authors = _format_authors_harvard(source.authors) if source.authors else ""
        year = f"({source.year})" if source.year else "(n.d.)"

        if source.source_type == "journal":
            title = f"'{source.title}',"
            journal = f" *{source.journal}*," if source.journal else ""
            vol = f" {source.volume}" if source.volume else ""
            issue = f"({source.issue})" if source.issue else ""
            pages = f", pp. {source.pages}." if source.pages else "."
            doi_url = f" doi: {source.doi}" if source.doi else (f" Available at: {source.url}" if source.url else "")
            accessed = f" (Accessed: {source.accessed_date})." if source.url and not source.doi else ""
            return f"{authors} {year} {title}{journal}{vol}{issue}{pages}{doi_url}{accessed}".strip()

        elif source.source_type == "book":
            title = f"*{source.title}*."
            publisher = f" {source.publisher}." if source.publisher else ""
            return f"{authors} {year} {title}{publisher}".strip()

        else:
            title = f"*{source.title}*."
            site = f" {source.publisher}." if source.publisher else ""
            url = f" Available at: {source.url}" if source.url else ""
            accessed = f" (Accessed: {source.accessed_date})." if source.url else ""
            return f"{authors} {year} {title}{site}{url}{accessed}".strip()

    # --- Chicago 17th (Author-Date) ---
    def _inline_chicago(self, source, page=None):
        if source.authors:
            last = source.authors[0].strip().split()[-1]
            if len(source.authors) > 3:
                name_part = f"{last} et al."
            elif len(source.authors) == 2:
                last2 = source.authors[1].strip().split()[-1]
                name_part = f"{last} and {last2}"
            elif len(source.authors) == 3:
                lasts = [a.strip().split()[-1] for a in source.authors]
                name_part = f"{lasts[0]}, {lasts[1]}, and {lasts[2]}"
            else:
                name_part = last
        else:
            title_short = source.title[:30] + "..." if len(source.title) > 30 else source.title
            name_part = f'"{title_short}"'
        year = source.year if source.year else "n.d."
        if page:
            return f"({name_part} {year}, {page})"
        return f"({name_part} {year})"

    def _ref_chicago(self, source):
        authors = _format_authors_chicago(source.authors) if source.authors else ""
        year = f"{source.year}." if source.year else "n.d."

        if source.source_type == "journal":
            title = f'"{source.title}."'
            journal = f" *{source.journal}*" if source.journal else ""
            vol = f" {source.volume}" if source.volume else ""
            issue = f", no. {source.issue}" if source.issue else ""
            year_part = f" ({source.year})" if source.year else ""
            pages = f": {source.pages}." if source.pages else "."
            doi_url = f" https://doi.org/{source.doi}." if source.doi else (f" {source.url}." if source.url else "")
            return f"{authors}. {year} {title}{journal}{vol}{issue}{year_part}{pages}{doi_url}".strip()

        elif source.source_type == "book":
            title = f"*{source.title}*."
            publisher = f" {source.publisher}." if source.publisher else ""
            return f"{authors}. {year} {title}{publisher}".strip()

        else:
            title = f'"{source.title}."'
            site = f" {source.publisher}." if source.publisher else ""
            url = f" {source.url}." if source.url else ""
            return f"{authors}. {year} {title}{site}{url}".strip()
