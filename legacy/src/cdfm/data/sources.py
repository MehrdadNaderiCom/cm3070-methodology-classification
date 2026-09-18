"""Clients for the two public interfaces the corpus is drawn from.

arXiv supplies Computer Science records. OpenAlex supplies Information Systems
and Information Technology records, which arXiv does not carry, and which the
project needs before a three way discipline classifier can mean anything.

Neither client stores full text. Both return titles, abstracts and metadata,
which is what the sources make available for indexing and what the project's
data boundary permits.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterator
from dataclasses import dataclass, field

USER_AGENT = "cdfm/0.1 (CM3070 final project; academic use)"

OPENALEX_ENDPOINT = "https://api.openalex.org/works"
ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

OPENALEX_PAGE_SIZE = 200
ARXIV_PAGE_SIZE = 100

OPENALEX_DELAY_SECONDS = 0.2
ARXIV_DELAY_SECONDS = 3.0

MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 5.0


@dataclass
class Paper:
    """One publication record, before any label is attached."""

    paper_id: str
    source: str
    title: str
    abstract: str
    year: int
    venue: str = ""
    venue_id: str = ""
    keywords: list[str] = field(default_factory=list)
    primary_category: str = ""
    topic: str = ""
    subfield: str = ""
    doi: str = ""

    def is_usable(self, minimum_abstract_chars: int = 300) -> bool:
        """Reject records too thin to carry methodological evidence.

        The threshold is not arbitrary. The shortest abstract in the frozen 50
        paper sample runs to 335 characters, so anything materially below that
        is outside the range the prototype was shown to work on.
        """
        return bool(self.title.strip()) and len(self.abstract.strip()) >= minimum_abstract_chars


def _request(url: str, timeout: int = 60) -> str:
    """Fetch a URL, retrying on transient failures with a widening delay."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"failed after {MAX_ATTEMPTS} attempts: {url}") from last_error


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Rebuild running text from OpenAlex's inverted index.

    OpenAlex stores abstracts as a word to positions mapping rather than as a
    string. Reversing it gives back the original word order.
    """
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, locations in inverted_index.items():
        for location in locations:
            positions[location] = word
    return " ".join(positions[index] for index in sorted(positions))


def normalise(text: str) -> str:
    return " ".join(text.split())


class OpenAlex:
    """Cursor paginated reader over the OpenAlex works endpoint.

    Args:
        mailto: Optional contact address. OpenAlex offers a faster shared pool
            to callers who identify themselves. Left unset by default so that no
            personal address is sent without a deliberate choice.
    """

    def __init__(self, mailto: str | None = None):
        self.mailto = mailto

    def _url(self, filters: str, cursor: str, per_page: int) -> str:
        parameters = {"filter": filters, "per-page": str(per_page), "cursor": cursor}
        if self.mailto:
            parameters["mailto"] = self.mailto
        return f"{OPENALEX_ENDPOINT}?{urllib.parse.urlencode(parameters, safe=':,|<>')}"

    def works(self, filters: str, limit: int) -> Iterator[dict]:
        """Yield raw work objects matching an OpenAlex filter expression."""
        cursor = "*"
        seen = 0
        while cursor and seen < limit:
            page_size = min(OPENALEX_PAGE_SIZE, limit - seen)
            payload = json.loads(self._request_page(filters, cursor, page_size))
            results = payload.get("results", [])
            if not results:
                return
            for work in results:
                yield work
                seen += 1
                if seen >= limit:
                    return
            cursor = payload.get("meta", {}).get("next_cursor")
            time.sleep(OPENALEX_DELAY_SECONDS)

    def _request_page(self, filters: str, cursor: str, per_page: int) -> str:
        return _request(self._url(filters, cursor, per_page))

    def count(self, filters: str) -> int:
        """Return how many works match, without paging through them."""
        payload = json.loads(self._request_page(filters, "*", 1))
        return int(payload.get("meta", {}).get("count", 0))

    def sample(self, filters: str, size: int, seed: int) -> list[dict]:
        """Draw a reproducible random sample rather than the first N results.

        Taking the first N returns whatever the default ordering puts at the
        front, which for a citation indexed source means the most cited and the
        oldest. That would bias the corpus towards a particular kind of paper
        before any modelling begins. OpenAlex supports seeded sampling, so the
        draw is random and still reproduces on a later run.
        """
        parameters = {
            "filter": filters,
            "sample": str(min(size, OPENALEX_PAGE_SIZE)),
            "seed": str(seed),
            "per-page": str(min(size, OPENALEX_PAGE_SIZE)),
        }
        if self.mailto:
            parameters["mailto"] = self.mailto
        url = f"{OPENALEX_ENDPOINT}?{urllib.parse.urlencode(parameters, safe=':,|<>')}"
        payload = json.loads(_request(url))
        time.sleep(OPENALEX_DELAY_SECONDS)
        return payload.get("results", [])

    @staticmethod
    def to_paper(work: dict) -> Paper:
        location = work.get("primary_location") or {}
        venue = location.get("source") or {}
        topic = work.get("primary_topic") or {}
        openalex_id = str(work.get("id", "")).rsplit("/", 1)[-1]

        return Paper(
            paper_id=openalex_id,
            source="OpenAlex",
            title=normalise(work.get("title") or work.get("display_name") or ""),
            abstract=normalise(reconstruct_abstract(work.get("abstract_inverted_index"))),
            year=int(work.get("publication_year") or 0),
            venue=venue.get("display_name") or "",
            venue_id=str(venue.get("id", "")).rsplit("/", 1)[-1],
            keywords=[k.get("display_name", "") for k in (work.get("keywords") or [])],
            topic=topic.get("display_name") or "",
            subfield=((topic.get("subfield") or {}).get("display_name") or ""),
            doi=str(work.get("doi") or "").replace("https://doi.org/", ""),
        )


class ArXiv:
    """Reader over the arXiv Atom interface.

    arXiv asks callers to leave three seconds between requests. That pacing is
    built in rather than left to the caller to remember.
    """

    def search(self, query: str, limit: int) -> Iterator[Paper]:
        """Yield papers matching an arXiv search expression, newest first."""
        fetched = 0
        while fetched < limit:
            page_size = min(ARXIV_PAGE_SIZE, limit - fetched)
            parameters = urllib.parse.urlencode(
                {
                    "search_query": query,
                    "start": fetched,
                    "max_results": page_size,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
            )
            body = _request(f"{ARXIV_ENDPOINT}?{parameters}")
            entries = ElementTree.fromstring(body).findall(f"{ATOM}entry")
            if not entries:
                return
            for entry in entries:
                yield self.to_paper(entry)
                fetched += 1
            time.sleep(ARXIV_DELAY_SECONDS)

    @staticmethod
    def to_paper(entry: ElementTree.Element) -> Paper:
        raw_id = entry.findtext(f"{ATOM}id", default="")
        published = entry.findtext(f"{ATOM}published", default="")
        category = entry.find(f"{ARXIV_NS}primary_category")
        return Paper(
            paper_id=raw_id.rsplit("/abs/", 1)[-1],
            source="arXiv",
            title=normalise(entry.findtext(f"{ATOM}title", default="")),
            abstract=normalise(entry.findtext(f"{ATOM}summary", default="")),
            year=int(published[:4]) if published[:4].isdigit() else 0,
            venue="arXiv",
            primary_category=category.get("term", "") if category is not None else "",
        )
