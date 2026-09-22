import logfire
from bs4 import BeautifulSoup


def parse_html_content(content: str):
    """Parses an HTML string using BeautifulSoup, cleans scripts/styles/meta,
    and extracts readable text for RAG. Shared by file parsing and web intake."""
    with logfire.span("📄 HTML Parsing"):
        soup = BeautifulSoup(content or "", "html.parser")

        # 1. Remove Junk (Scripts, Styles, Metadata)
        for script in soup(["script", "style", "meta", "noscript"]):
            script.decompose()

        # 2. Extract Text
        text = soup.get_text(separator="\n")

        # 3. Clean Whitespace (Collapse multiple newlines)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)


def parse_html(file_path: str):
    """
    Parses HTML content using BeautifulSoup.
    Cleans scripts, styles, and extracts readable text for RAG.
    """
    with logfire.span("📄 HTML Parsing", filename=file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return parse_html_content(content)
        except Exception as e:
            logfire.error(f"❌ HTML Parse Failed: {e}")
            raise e



