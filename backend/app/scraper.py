"""
scraper.py
----------
Web scraping module (Unit 3: Web Scraping using requests + BeautifulSoup).

Fetches reference/"normal range" data for common vitals and lab tests so
imported patient data can be automatically flagged against up-to-date
clinical reference ranges, instead of hardcoding thresholds everywhere.

Works against any HTML page with a <table> of "test / low / high / unit"
data. `scrape_reference_ranges_from_url()` hits a live URL with `requests`;
`scrape_reference_ranges_from_html()` parses an HTML string directly and is
used by the offline demo / unit tests (handy when the target site is
unreachable in a sandboxed environment, or for fast, deterministic tests).
"""

from __future__ import annotations
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from app.exceptions import FileProcessingError


@dataclass
class ReferenceRange:
    test_name: str
    low: float
    high: float
    unit: str


def _parse_table(soup: BeautifulSoup) -> list[ReferenceRange]:
    ranges: list[ReferenceRange] = []
    table = soup.find("table", {"id": "reference-ranges"}) or soup.find("table")
    if table is None:
        return ranges

    rows = table.find_all("tr")
    for row in rows[1:]:  # skip header
        cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) < 4:
            continue
        name, low, high, unit = cells[0], cells[1], cells[2], cells[3]
        try:
            ranges.append(ReferenceRange(test_name=name, low=float(low), high=float(high), unit=unit))
        except ValueError:
            continue  # skip malformed rows rather than failing the whole scrape
    return ranges


def scrape_reference_ranges_from_html(html: str) -> list[ReferenceRange]:
    soup = BeautifulSoup(html, "html.parser")
    return _parse_table(soup)


def scrape_reference_ranges_from_url(url: str, timeout: int = 10) -> list[ReferenceRange]:
    """Fetch and parse a live reference-ranges page. Requires outbound
    network access to `url`, which may be restricted in sandboxed/offline
    environments — use scrape_reference_ranges_from_html() there instead."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "MediSenseAI-Bot/1.0"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise FileProcessingError(f"Failed to fetch reference data from {url}: {exc}") from exc
    return scrape_reference_ranges_from_html(resp.text)


# A small offline fixture standing in for a real public reference-ranges
# page (e.g. a hospital/lab site's "normal values" table), so the demo and
# tests are deterministic and don't depend on network access.
SAMPLE_REFERENCE_HTML = """
<html><body>
<table id="reference-ranges">
  <tr><th>Test</th><th>Low</th><th>High</th><th>Unit</th></tr>
  <tr><td>Fasting Glucose</td><td>70</td><td>100</td><td>mg/dL</td></tr>
  <tr><td>Cholesterol</td><td>125</td><td>200</td><td>mg/dL</td></tr>
  <tr><td>Heart Rate</td><td>60</td><td>100</td><td>bpm</td></tr>
  <tr><td>SpO2</td><td>95</td><td>100</td><td>%</td></tr>
  <tr><td>HbA1c</td><td>4.0</td><td>5.6</td><td>%</td></tr>
</table>
</body></html>
"""


if __name__ == "__main__":
    ranges = scrape_reference_ranges_from_html(SAMPLE_REFERENCE_HTML)
    print(f"Parsed {len(ranges)} reference ranges:")
    for r in ranges:
        print(f"  {r.test_name}: {r.low}-{r.high} {r.unit}")
