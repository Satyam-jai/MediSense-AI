"""Tests for Unit 3: bulk import and scraping."""
from app.data_import import generate_sample_workbook, import_patients_from_file
from app.scraper import scrape_reference_ranges_from_html, SAMPLE_REFERENCE_HTML


def test_generate_and_import_sample_workbook(tmp_path):
    workbook = generate_sample_workbook(tmp_path / "sample.xlsx", n=10)
    report = import_patients_from_file(workbook)
    # 10 good rows + 1 intentionally broken row baked into the generator
    assert report.success_count == 10
    assert report.error_count == 1
    assert "Row" in report.errors[0]


def test_scraper_parses_offline_fixture():
    ranges = scrape_reference_ranges_from_html(SAMPLE_REFERENCE_HTML)
    names = {r.test_name for r in ranges}
    assert "Fasting Glucose" in names
    assert "SpO2" in names
    glucose = next(r for r in ranges if r.test_name == "Fasting Glucose")
    assert glucose.low == 70 and glucose.high == 100
