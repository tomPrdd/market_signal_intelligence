"""One-shot curation of new_doc/ into data/raw/<TICKER>/management/.

Maps each hand-collected file to a dated, titled filename. Publication dates
come from the document text when printed (results releases), otherwise from
filing conventions (a FY-N 20-F is filed ~late March N+1; half-year reports
~late July; URDs ~mid-March N+1). Skips >60MB files (parser-hostile), non-PDF
junk, and documents duplicating an existing English version.

Also renames the pre-existing TTE.PA management PDFs (fetched earlier from the
IR page with undated CDN names) onto the same convention.
"""

import shutil
from pathlib import Path

NEW = Path("new_doc")
RAW = Path("data/raw")

# (source folder, source filename) -> (ticker, target filename) ; None = skip
MAPPING: dict[tuple[str, str], tuple[str, str] | None] = {
    # ---------------- L'Oreal -> OR.PA ----------------
    ("L'Oreal", "LOREAL_AnnualResults_2018_EN.pdf"): (
        "OR.PA",
        "2019-02-07-annual-results-2018.pdf",
    ),
    ("L'Oreal", "LOREAL_Annual_Results_2019_EN.pdf"): (
        "OR.PA",
        "2020-02-06-annual-results-2019.pdf",
    ),
    ("L'Oreal", "LOREAL_Annual_Results_2019_EN-2.pdf"): None,  # exact duplicate
    ("L'Oreal", "LOREAL_AnnualResults_2021_EN.pdf"): (
        "OR.PA",
        "2022-02-09-annual-results-2021.pdf",
    ),
    ("L'Oreal", "CPFY25ENv12.02 17.00.pdf"): ("OR.PA", "2026-02-12-annual-results-2025.pdf"),
    ("L'Oreal", "CP_1Q26_EN v22.04.26.pdf"): ("OR.PA", "2026-04-22-first-quarter-2026-sales.pdf"),
    ("L'Oreal", "LOREAL_HALF-YEAR_REPORT_2018.pdf"): (
        "OR.PA",
        "2018-07-25-half-year-financial-report-2018.pdf",
    ),
    ("L'Oreal", "LOREAL_2019_Half-Year_report_EN.pdf"): (
        "OR.PA",
        "2019-07-25-half-year-financial-report-2019.pdf",
    ),
    ("L'Oreal", "5.Consolidated and annual financial statements of 2019.pdf"): (
        "OR.PA",
        "2020-03-20-consolidated-financial-statements-2019.pdf",
    ),
    ("L'Oreal", "4. Consolidated Financial Statements.pdf"): (
        "OR.PA",
        "2021-03-20-consolidated-financial-statements-2020.pdf",
    ),
    ("L'Oreal", "4 - 2021 consolidated financial statements.pdf"): (
        "OR.PA",
        "2022-03-20-consolidated-financial-statements-2021.pdf",
    ),
    ("L'Oreal", "Chap 5 LOREAL_DEU_2022 ENG.pdf"): (
        "OR.PA",
        "2023-03-20-consolidated-financial-statements-2022.pdf",
    ),
    ("L'Oreal", "15-Chap 5 LOREAL_DEU_2023 ENG.pdf"): (
        "OR.PA",
        "2024-03-20-consolidated-financial-statements-2023.pdf",
    ),
    ("L'Oreal", "Consolidated financial statements.pdf"): (
        "OR.PA",
        "2025-03-20-consolidated-financial-statements-2024.pdf",
    ),
    ("L'Oreal", "LOREAL2020AnnualReport.pdf"): ("OR.PA", "2021-04-01-annual-report-2020.pdf"),
    ("L'Oreal", "files.pdf"): ("OR.PA", "2024-04-01-rapport-annuel-2023-fr.pdf"),
    ("L'Oreal", "LOREAL2024AnnualReport.pdf"): ("OR.PA", "2025-04-01-annual-report-2024.pdf"),
    # ---------------- LVMH -> MC.PA ----------------
    ("LVMH", "rapport-annuel-lvmh-2018_va.pdf"): ("MC.PA", "2019-04-01-annual-report-2018.pdf"),
    ("LVMH", "lvmh_rapport-annuel-2019_gb.pdf"): ("MC.PA", "2020-04-01-annual-report-2019.pdf"),
    ("LVMH", "lvmh_rapport-annuel-2020-va.pdf"): ("MC.PA", "2021-04-01-annual-report-2020.pdf"),
    ("LVMH", "lvmh_rapport-annuel-2021-va.pdf"): ("MC.PA", "2022-04-01-annual-report-2021.pdf"),
    ("LVMH", "Zn1lix5LeNNTwmKk_lvmh_2022_annual-report.pdf"): (
        "MC.PA",
        "2023-04-01-annual-report-2022.pdf",
    ),
    ("LVMH", "ZlXX6aWtHYXtT2hB_lvmh_2023-annual-report-1-.pdf"): (
        "MC.PA",
        "2024-04-01-annual-report-2023.pdf",
    ),
    ("LVMH", "Z-Qi1HdAxsiBv78A_LVMH_2024AnnualReport.pdf"): (
        "MC.PA",
        "2025-04-01-annual-report-2024.pdf",
    ),
    ("LVMH", "ac0pyZGXnQHGZK4S_LVMH_RA2025_GB_MEL1.pdf"): (
        "MC.PA",
        "2026-04-01-annual-report-2025.pdf",
    ),
    ("LVMH", "rapportfinanciersemestriel2019_va.pdf"): (
        "MC.PA",
        "2019-07-24-interim-financial-report-h1-2019.pdf",
    ),
    ("LVMH", "lvmh-rapport-financier-semestriel-2020-va.pdf"): (
        "MC.PA",
        "2020-07-27-interim-financial-report-h1-2020.pdf",
    ),
    ("LVMH", "presentation-hy-20-va.pdf"): (
        "MC.PA",
        "2020-07-27-first-half-2020-results-presentation.pdf",
    ),
    ("LVMH", "ZoRebB5LeNNTwutQ_lvmh_interim-financial-report-2022.pdf"): (
        "MC.PA",
        "2022-07-26-interim-financial-report-h1-2022.pdf",
    ),
    ("LVMH", "lvmh_2023-first-half-financial-report.pdf"): (
        "MC.PA",
        "2023-07-25-interim-financial-report-h1-2023.pdf",
    ),
    ("LVMH", "Zp_G9x5LeNNTxbfv_LVMH_2024Firsthalffinancialreport.pdf"): (
        "MC.PA",
        "2024-07-23-interim-financial-report-h1-2024.pdf",
    ),
    ("LVMH", "aIJQNlGsbswqTOVX_LVMH_2025Firsthalffinancialreport.pdf"): (
        "MC.PA",
        "2025-07-22-interim-financial-report-h1-2025.pdf",
    ),
    ("LVMH", "financial-documents-december-31-2021.pdf"): (
        "MC.PA",
        "2022-01-27-financial-documents-fy-2021.pdf",
    ),
    ("LVMH", "financial-documents-lvmh-december-31-2023.pdf"): (
        "MC.PA",
        "2024-01-25-financial-documents-fy-2023.pdf",
    ),
    ("LVMH", "Z5kVBpbqstJ999KR_Financialdocuments-December31%2C2024.pdf"): (
        "MC.PA",
        "2025-01-28-financial-documents-fy-2024.pdf",
    ),
    ("LVMH", "Z6ysCJbqstJ9-hws_LVMHComptesconsolid%C3%A9s2024-VA-.pdf"): (
        "MC.PA",
        "2025-01-28-consolidated-financial-statements-2024.pdf",
    ),
    ("LVMH", "aY242t0YXLCxVuF9_LVMHComptesconsolid%C3%A9s2025-VA--VDEFpubli%C3%A9e.pdf"): (
        "MC.PA",
        "2026-01-27-consolidated-financial-statements-2025.pdf",
    ),
    # ---------------- Sanofi -> SAN.PA ----------------
    ("Sanofi", "2018-01-01-form-20-f-2018-en.pdf"): ("SAN.PA", "2019-03-05-form-20-f-2018.pdf"),
    ("Sanofi", "2019-01-01-form-20-f-2019-en.pdf"): ("SAN.PA", "2020-03-03-form-20-f-2019.pdf"),
    ("Sanofi", "2020-01-01-form-20-f-2020-en.pdf"): ("SAN.PA", "2021-03-03-form-20-f-2020.pdf"),
    ("Sanofi", "2021-01-01-form-20-f-2021-en.pdf"): ("SAN.PA", "2022-03-03-form-20-f-2021.pdf"),
    ("Sanofi", "2022-01-01-form-20-f-2022-en.pdf"): ("SAN.PA", "2023-03-03-form-20-f-2022.pdf"),
    ("Sanofi", "2023-01-01-form-20-f-2023-en.pdf"): ("SAN.PA", "2024-03-01-form-20-f-2023.pdf"),
    ("Sanofi", "2024-01-01-form-20-f-2024-en.pdf"): ("SAN.PA", "2025-03-01-form-20-f-2024.pdf"),
    ("Sanofi", "2025-01-01-form-20-f-2025-en.pdf"): ("SAN.PA", "2026-03-01-form-20-f-2025.pdf"),
    ("Sanofi", "SWI_Sanofi_Amendement_20-F_2025_VMEL.pdf"): (
        "SAN.PA",
        "2026-04-01-form-6-k-amendment-20-f-2025.pdf",
    ),
    ("Sanofi", "2018-01-01-document-denregistrement-universel-2018-fr.pdf"): (
        "SAN.PA",
        "2019-03-20-document-enregistrement-universel-2018-fr.pdf",
    ),
    ("Sanofi", "2018-01-01-sanofi-half-year-financial-report-2018-en.pdf"): (
        "SAN.PA",
        "2018-07-31-half-year-financial-report-2018.pdf",
    ),
    ("Sanofi", "2019-01-01-sanofi-half-year-financial-report-2019-en.pdf"): (
        "SAN.PA",
        "2019-07-29-half-year-financial-report-2019.pdf",
    ),
    ("Sanofi", "2020-01-01-sanofi-half-year-financial-report-2020-fr.pdf"): (
        "SAN.PA",
        "2020-07-29-rapport-financier-semestriel-2020-fr.pdf",
    ),
    ("Sanofi", "2021-01-01-sanofi-half-year-financial-report-2021-en.pdf"): (
        "SAN.PA",
        "2021-07-29-half-year-financial-report-2021.pdf",
    ),
    ("Sanofi", "2022-01-01-sanofi-half-year-financial-report-2022-en-01.pdf"): (
        "SAN.PA",
        "2022-07-28-half-year-financial-report-2022.pdf",
    ),
    ("Sanofi", "2023-sanofi-rapport-financier-semestriel.pdf"): (
        "SAN.PA",
        "2023-07-28-rapport-financier-semestriel-2023-fr.pdf",
    ),
    ("Sanofi", "sanofi-half-year-financial-report-2024.pdf"): (
        "SAN.PA",
        "2024-07-25-half-year-financial-report-2024.pdf",
    ),
    ("Sanofi", "Half-year-financial-report-2025.pdf"): (
        "SAN.PA",
        "2025-07-31-half-year-financial-report-2025.pdf",
    ),
    ("Sanofi", "2019-01-01-sanofi-integrated-report-en.pdf"): (
        "SAN.PA",
        "2020-05-01-integrated-report-2019.pdf",
    ),
    ("Sanofi", "2020-01-01-sanofi-Integrated-report-en.pdf"): (
        "SAN.PA",
        "2021-05-01-integrated-report-2020.pdf",
    ),
    ("Sanofi", "SANOFI-Integrated-Annual-Report-2022-EN.pdf"): (
        "SAN.PA",
        "2023-05-01-integrated-annual-report-2022.pdf",
    ),
    ("Sanofi", "2022-01-01-declaration-of-extra-financial-performance-en.pdf"): (
        "SAN.PA",
        "2023-03-20-declaration-extra-financial-performance-2022.pdf",
    ),
    ("Sanofi", "2023-01-01-declaration-of-extra-financial-performance-en.pdf"): (
        "SAN.PA",
        "2024-03-20-declaration-extra-financial-performance-2023.pdf",
    ),
    # ---------------- TotalEnergies -> TTE.PA (new files only) ----------------
    # Skips: FR duplicates of English docs already in data/raw/TTE.PA, and >60MB files.
    ("TotalEnergies", "2017-form-20-f-web.pdf"): None,  # already in data/raw
    ("TotalEnergies", "2018-form-20-f-web.pdf"): None,  # already in data/raw
    ("TotalEnergies", "2019-total-form-20-f.pdf"): None,  # already in data/raw
    ("TotalEnergies", "2020-total-form-20-f.pdf"): None,  # already in data/raw
    ("TotalEnergies", "Form_20-F_2021.pdf"): None,  # already in data/raw
    ("TotalEnergies", "ddr2017-fr.pdf"): None,  # EN version already in data/raw
    ("TotalEnergies", "ddr2018-fr.pdf"): None,  # EN version already in data/raw
    ("TotalEnergies", "total_document_enregistrement_universel_2019.pdf"): None,  # EN in data/raw
    ("TotalEnergies", "document-enregistrement-universel-2020.pdf"): None,  # EN in data/raw
    ("TotalEnergies", "files.pdf"): None,  # URD 2021 FR; EN (deu-21-va) in data/raw
    ("TotalEnergies", "TotalEnergies_DEU_2022_VF.pdf"): None,  # EN urd-2022 in data/raw
    (
        "TotalEnergies",
        "totalenergies_document-enregistrement-universel-2023_2023_fr_pdf.pdf",
    ): None,  # EN in data/raw
    (
        "TotalEnergies",
        "totalenergies_document-enregistrement-universel-2024_2025_fr.pdf",
    ): None,  # EN in data/raw
    (
        "TotalEnergies",
        "totalenergies_document-enregistrement-universel-2025_2026_fr.pdf",
    ): None,  # 63MB > cap
    ("TotalEnergies", "TotalEnergies_Form_20-F_2022.pdf"): None,  # 106MB > cap
    ("TotalEnergies", "totalenergies_form-20-f-2023_2023_en_pdf.pdf"): None,  # 110MB > cap
    ("TotalEnergies", "totalenergies_form-20-f-2024_2025_en.pdf"): None,  # 108MB > cap
    ("TotalEnergies", "totalenergies_form-20-f-2025_2026_en.pdf"): None,  # 151MB + garbled text
    (
        "TotalEnergies",
        "totalenergies_sustainability-climate-2026-progress-report_2026_fr.pdf",
    ): None,  # 111MB > cap
    ("TotalEnergies", "factbook-2017_web_0.pdf"): ("TTE.PA", "2018-04-15-factbook-2017.pdf"),
    ("TotalEnergies", "factbook_2018.pdf"): ("TTE.PA", "2019-04-15-factbook-2018.pdf"),
    ("TotalEnergies", "Factbook_2019.pdf"): ("TTE.PA", "2020-04-15-factbook-2019.pdf"),
    ("TotalEnergies", "Factbook_2020.pdf"): ("TTE.PA", "2021-04-15-factbook-2020.pdf"),
    ("TotalEnergies", "Factbook_2021.pdf"): ("TTE.PA", "2022-04-15-factbook-2021.pdf"),
    ("TotalEnergies", "Factbook_2022.pdf"): ("TTE.PA", "2023-04-15-factbook-2022.pdf"),
    ("TotalEnergies", "totalenergies_factbook-2023_2024_en_pdf.pdf"): (
        "TTE.PA",
        "2024-04-15-factbook-2023.pdf",
    ),
    ("TotalEnergies", "totalenergies_factbook-2024_2025_en.pdf"): (
        "TTE.PA",
        "2025-04-15-factbook-2024.pdf",
    ),
    ("TotalEnergies", "totalenergies_factbook-2025_2026_en.pdf"): (
        "TTE.PA",
        "2026-04-15-factbook-2025.pdf",
    ),
    ("TotalEnergies", "total_climat_2018.pdf"): ("TTE.PA", "2018-09-01-rapport-climat-2018-fr.pdf"),
    ("TotalEnergies", "total_rapport_climat_2019.pdf"): (
        "TTE.PA",
        "2019-11-01-rapport-climat-2019-fr.pdf",
    ),
    ("TotalEnergies", "TOTAL_RAPPORT_CLIMAT_2020.pdf"): (
        "TTE.PA",
        "2020-09-01-rapport-climat-2020-fr.pdf",
    ),
    ("TotalEnergies", "Sustainability_Climate_2022_Progress_Report_version_accessible_FR.pdf"): (
        "TTE.PA",
        "2022-03-15-sustainability-climate-progress-report-2022-fr.pdf",
    ),
    ("TotalEnergies", "Sustainability_Climate_2023_Progress_Report_FR.pdf"): (
        "TTE.PA",
        "2023-05-15-sustainability-climate-progress-report-2023-fr.pdf",
    ),
    (
        "TotalEnergies",
        "totalenergies_sustainability-climate-2024-progress-report_2024_fr_pdf.pdf",
    ): ("TTE.PA", "2024-03-15-sustainability-climate-progress-report-2024-fr.pdf"),
    ("TotalEnergies", "totalenergies_sustainability-climate-2025-progress-report_2025_fr.pdf"): (
        "TTE.PA",
        "2025-03-15-sustainability-climate-progress-report-2025-fr.pdf",
    ),
}

# Existing TTE.PA management files (IR fetch) -> dated names on the same convention
TTE_EXISTING_RENAMES = {
    "total-form-20-f-2011.pdf": "2012-03-25-form-20-f-2011.pdf",
    "form-20-f-0.pdf": "2013-03-25-form-20-f-2012.pdf",
    "form-20-f-2013.pdf": "2014-03-25-form-20-f-2013.pdf",
    "form-20-f-2015-web-version.pdf": "2016-03-25-form-20-f-2015.pdf",
    "2016-form-20-f-web-0.pdf": "2017-03-25-form-20-f-2016.pdf",
    "2017-form-20-f-web.pdf": "2018-03-25-form-20-f-2017.pdf",
    "2018-form-20-f-web.pdf": "2019-03-25-form-20-f-2018.pdf",
    "2019-total-form-20-f.pdf": "2020-03-25-form-20-f-2019.pdf",
    "2020-total-form-20-f.pdf": "2021-03-25-form-20-f-2020.pdf",
    "form-20-f-2021.pdf": "2022-03-25-form-20-f-2021.pdf",
    "total-registration-document-2011.pdf": "2012-03-20-registration-document-2011.pdf",
    "total-docref-2012-va.pdf": "2013-03-20-registration-document-2012.pdf",
    "registration-document-2013.pdf": "2014-03-20-registration-document-2013.pdf",
    "registration-document-v3-2014.pdf": "2015-03-20-registration-document-2014.pdf",
    "registration-document-2015.pdf": "2016-03-20-registration-document-2015.pdf",
    "ddr2016-va-web-1.pdf": "2017-03-20-registration-document-2016.pdf",
    "ddr2017-en-accessible.pdf": "2018-03-20-registration-document-2017.pdf",
    "ddr2018-en.pdf": "2019-03-20-registration-document-2018.pdf",
    "2019-total-universal-registration-document.pdf": "2020-03-20-universal-registration-document-2019.pdf",
    "2020-universal-registration-document.pdf": "2021-03-20-universal-registration-document-2020.pdf",
    "deu-21-va.pdf": "2022-03-20-universal-registration-document-2021.pdf",
    "totalenergies-urd-2022-en.pdf": "2023-03-20-universal-registration-document-2022.pdf",
    "totalenergies-universal-registration-document-2023-2023-en-pdf.pdf": "2024-03-20-universal-registration-document-2023.pdf",
    "totalenergies-universal-registration-document-2024-2025-en.pdf": "2025-03-20-universal-registration-document-2024.pdf",
    "dvpvilvkarszvhoa8u0fr5mx-financial-information-at-september-30-2025.pdf": "2025-10-30-financial-information-q3-2025.pdf",
}


def main() -> None:
    copied, skipped, missing = 0, 0, 0
    for (folder, fname), target in MAPPING.items():
        src = NEW / folder / fname
        if not src.exists():
            print(f"  !! missing: {src}")
            missing += 1
            continue
        if target is None:
            skipped += 1
            continue
        ticker, new_name = target
        dest = RAW / ticker / "management" / new_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
            copied += 1

    renamed = 0
    tte_dir = RAW / "TTE.PA" / "management"
    for old, new in TTE_EXISTING_RENAMES.items():
        src = tte_dir / old
        if src.exists():
            src.rename(tte_dir / new)
            renamed += 1

    print(f"copied {copied}, skipped {skipped}, missing {missing}, renamed existing TTE {renamed}")
    for ticker in ("OR.PA", "MC.PA", "SAN.PA", "TTE.PA"):
        d = RAW / ticker / "management"
        n = len(list(d.glob("*.pdf"))) if d.exists() else 0
        print(f"  {ticker}: {n} management PDFs")


if __name__ == "__main__":
    main()
