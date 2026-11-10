"""
demo_phase2.py
---------------
Runnable demo of the Unit 3 layer: bulk Excel import + reference-range
scraping, independent of the Flask server / dashboard.

Run with:
    cd backend
    python demo_phase2.py
"""

from pathlib import Path

from app.data_import import generate_sample_workbook, import_patients_from_file
from app.scraper import scrape_reference_ranges_from_html, SAMPLE_REFERENCE_HTML


def main():
    print("--- Generating a sample patient workbook (openpyxl) ---")
    workbook_path = generate_sample_workbook("data/sample_patients.xlsx", n=12)
    print("Wrote:", workbook_path)

    print("\n--- Bulk-importing patients (pandas cleaning + validation) ---")
    report = import_patients_from_file(workbook_path)
    print(f"Imported {report.success_count} patients, {report.error_count} failed")
    for err in report.errors:
        print("  ERROR:", err)
    for p in report.patients[:3]:
        print(" ", p.name, "-", [r.summary() for r in p.records])

    print("\n--- Scraping reference ranges (BeautifulSoup) ---")
    ranges = scrape_reference_ranges_from_html(SAMPLE_REFERENCE_HTML)
    for r in ranges:
        print(f"  {r.test_name}: {r.low}-{r.high} {r.unit}")


if __name__ == "__main__":
    main()
