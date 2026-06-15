#!/usr/bin/env python3
"""
rename_questions.py — Task 03: Fix a Broken Workflow

The actual problem was never the files — the ZIP had all images correctly,
and the Excel had the mapping from random filenames to question numbers.
Nobody had connected the two. This script does that automatically.

Output: Q1.png, S1.png, Q2.png, S2.png ... without anyone taking a screenshot.

Usage:
    python scripts/rename_questions.py
    python scripts/rename_questions.py --zip input/questions.zip --excel input/metadata.xlsx
    python scripts/rename_questions.py --zip input/questions.zip --excel input/metadata.xlsx --output output/renamed/
"""

import argparse
import os
import re
import sys
import zipfile

import pandas as pd


def _is_numeric_column(df, col):
    values = [str(x).strip() for x in df[col].dropna()]
    if not values:
        return False
    numeric = sum(1 for v in values if re.fullmatch(r"\d+", v))
    return numeric >= max(1, len(values) // 2)


def find_mapping_columns(df):
    """
    Auto-detect filename and question-number columns from Excel headers.

    We do this because different batches use different column names —
    one file says "Question Image", another says "filename", another says "file".
    Regex patterns catch all common variations so the script doesn't break
    every time the column name changes slightly.
    """
    # Defensive: if no dataframe provided, return no columns
    if df is None:
        return None, None, None

    filename_patterns = [
        r"question\s*image", r"question\s*img", r"question.*image",
        r"file\s*name", r"image\s*name", r"image", r"file",
        r"source", r"path", r"attachment",
    ]
    sol_filename_patterns = [
        r"sol\s*image", r"solution\s*image", r"answer\s*image",
        r"sol\s*img", r"solution.*image",
    ]
    question_patterns = [
        r"display\s*order", r"question\s*(no|num|number|id)?", r"q\s*\.?\s*(no|num|number|id)?",
        r"sr\s*\.?\s*(no|num)?", r"serial", r"number", r"id", r"sl\s*\.?\s*no",
    ]

    cols       = list(df.columns)
    cols_lower = [str(c).lower().strip() for c in cols]

    filename_col = None
    sol_filename_col = None
    question_col = None

    for pat in filename_patterns:
        for i, c in enumerate(cols_lower):
            if re.search(pat, c):
                filename_col = cols[i]
                break
        if filename_col:
            break

    for pat in sol_filename_patterns:
        for i, c in enumerate(cols_lower):
            if re.search(pat, c):
                sol_filename_col = cols[i]
                break
        if sol_filename_col:
            break

    for pat in question_patterns:
        for i, c in enumerate(cols_lower):
            if re.search(pat, c):
                question_col = cols[i]
                break
        if question_col:
            break

    if question_col and not _is_numeric_column(df, question_col):
        for i, c in enumerate(cols_lower):
            if _is_numeric_column(df, cols[i]):
                question_col = cols[i]
                break

    if question_col is None:
        for i, c in enumerate(cols_lower):
            if _is_numeric_column(df, cols[i]):
                question_col = cols[i]
                break

    return filename_col, sol_filename_col, question_col


def detect_type(filename, row_data=None):
    """
    Decide whether a file is a Question (Q) or Solution (S).

    We check the filename prefix first — QUES_ means question, SOLU_ means solution.
    This matches the actual PW file naming convention discovered during Task 3.
    Generic keywords like 'sol', 'ans' are checked as fallback for other formats.
    """
    name_lower = filename.lower()

    # PW-specific prefixes — most reliable signal
    if name_lower.startswith("solu_") or name_lower.startswith("sol_"):
        return "S"
    if name_lower.startswith("ques_") or name_lower.startswith("que_"):
        return "Q"

    # Generic keywords
    if any(kw in name_lower for kw in ["sol", "ans", "solution", "answer"]):
        return "S"
    if any(kw in name_lower for kw in ["que", "question", "ques"]):
        return "Q"

    # Check row data for a type column
    if row_data is not None:
        for col in row_data.index:
            val = str(row_data[col]).lower().strip()
            if val in ("question", "q", "que"):
                return "Q"
            if val in ("solution", "s", "sol", "answer", "ans"):
                return "S"

    return None


def extract_question_number(value):
    """
    Pull a number out of whatever format the cell uses.
    Handles: '1', 'Q1', 'Q.1', 'Question 1', '01' etc.
    """
    match = re.search(r"(\d+)", str(value).strip())
    return int(match.group(1)) if match else None


def rename_questions(zip_path, excel_path, output_dir):
    """
    Core logic: read Excel, extract ZIP, rename files.

    FIX: If Sol Image column is empty (as seen in real PW data),
    derive solution filename by replacing QUES_ prefix with SOLU_.
    The random string after the prefix is always the same for a question
    and its solution — this pattern was confirmed by inspecting the actual ZIP.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = None
    if excel_path:
        # Read Excel
        print(f"  Reading: {excel_path}")
        try:
            xls = pd.ExcelFile(excel_path)
        except Exception as exc:
            print(f"  ERROR: Failed to open Excel: {exc}")
            return []
        for sheet in xls.sheet_names:
            candidate = pd.read_excel(xls, sheet_name=sheet)
            if len(candidate) > 0 and len(candidate.columns) >= 2:
                df = candidate
                print(f"  Sheet: '{sheet}' — {len(df)} rows, {len(df.columns)} cols")
                break

        if df is None:
            print("  ERROR: No usable sheet found in Excel.")
            return []
    else:
        print("  No Excel provided: running in inference-only mode (heuristics will be used)")

    # Detect columns and build mapping only when we have Excel data
    file_to_info = {}
    if df is not None:
        filename_col, sol_filename_col, question_col = find_mapping_columns(df)

        if not filename_col:
            filename_col = df.columns[0]
            print(f"  WARNING: Could not detect question filename column — using '{filename_col}'")
        if not sol_filename_col:
            print("  WARNING: Could not detect solution filename column — will infer from question names if needed")
        if not question_col:
            question_col = df.columns[1] if len(df.columns) >= 2 else None
            print(f"  WARNING: Could not detect question ID column — using '{question_col}'")

        print(f"  Question filename col : '{filename_col}'")
        print(f"  Solution filename col : '{sol_filename_col}'")
        print(f"  Question col          : '{question_col}'")

        # Build lookup: original filename -> (question_number, type)
        for _, row in df.iterrows():
            q_num  = extract_question_number(row[question_col]) if question_col else None

            # Map question images
            qname = str(row[filename_col]).strip() if filename_col in row else ""
            if qname and qname.lower() != "nan":
                file_to_info[qname] = (q_num, "Q")

            # Map solution images if present
            if sol_filename_col and sol_filename_col in row:
                sname = str(row[sol_filename_col]).strip()
                if sname and sname.lower() != "nan":
                    file_to_info[sname] = (q_num, "S")

        print(f"  Mappings found: {len(file_to_info)}")
    else:
        print("  No Excel mappings available — proceeding with filename heuristics only")

    # Extract ZIP and build filename lookup
    if not zipfile.is_zipfile(zip_path):
        print(f"  ERROR: Not a valid ZIP: {zip_path}")
        return []

    print(f"  Extracting: {zip_path}")
    results   = []
    unmatched = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        image_files = [
            f for f in zf.namelist()
            if not f.startswith("__MACOSX")
            and not f.startswith(".")
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"))
        ]
        print(f"  Images in ZIP: {len(image_files)}")

        # Build a flat lookup: basename -> full zip path
        zip_lookup = {os.path.basename(p): p for p in image_files}

        q_counter = 1
        s_counter = 1

        for img_name, zip_path_inner in sorted(zip_lookup.items()):
            ext     = os.path.splitext(img_name)[1].lower() or ".png"
            matched = False

            for meta_name, (q_num, ftype) in file_to_info.items():
                if img_name.lower() == meta_name.lower() \
                        or os.path.splitext(img_name)[0].lower() \
                        == os.path.splitext(meta_name)[0].lower():

                    ftype  = ftype or "Q"
                    if q_num is None:
                        q_num = q_counter if ftype == "Q" else s_counter
                        if ftype == "Q": q_counter += 1
                        else:            s_counter += 1

                    new_name = f"{ftype}{q_num}{ext}"
                    matched  = True
                    break

            if not matched:
                # FIX: try deriving solution name from question name
                # e.g. SOLU_ENG_abc.png -> look for QUES_ENG_abc.png in metadata
                derived_q_name = img_name.replace("SOLU_ENG_", "QUES_ENG_") \
                                         .replace("SOL_ENG_",  "QUES_ENG_")
                if derived_q_name in file_to_info:
                    q_num, _ = file_to_info[derived_q_name]
                    new_name  = f"S{q_num}{ext}"
                    matched   = True

            if not matched:
                ftype    = detect_type(img_name) or "Q"
                q_num_fn = extract_question_number(img_name)
                if ftype == "S" and q_num_fn:
                    new_name = f"S{q_num_fn}{ext}"
                elif ftype == "Q" and q_num_fn:
                    new_name = f"Q{q_num_fn}{ext}"
                elif ftype == "S":
                    new_name = f"S{s_counter}{ext}"; s_counter += 1
                else:
                    new_name = f"Q{q_counter}{ext}"; q_counter += 1
                unmatched.append(img_name)

            # Handle duplicates
            out_path = os.path.join(output_dir, new_name)
            if os.path.exists(out_path):
                base, ext_p = os.path.splitext(new_name)
                i = 2
                while os.path.exists(os.path.join(output_dir, f"{base}_{i}{ext_p}")):
                    i += 1
                new_name = f"{base}_{i}{ext_p}"
                out_path = os.path.join(output_dir, new_name)

            with zf.open(zip_path_inner) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

            results.append((img_name, new_name))

    # Summary
    print(f"\n  Done!")
    print(f"  Total processed     : {len(results)}")
    print(f"  Matched via metadata: {len(results) - len(unmatched)}")
    print(f"  Matched via fallback: {len(unmatched)}")
    print(f"  Output              : {os.path.abspath(output_dir)}")
    print()
    for orig, new in sorted(results, key=lambda x: x[1]):
        print(f"    {orig:45s} -> {new}")

    if unmatched:
        print(f"\n  Files matched via heuristics (not in metadata):")
        for name in unmatched:
            print(f"    - {name}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Rename question/solution images using Excel metadata"
    )
    parser.add_argument("--zip",    default="input/questions.zip")
    parser.add_argument("--excel",  default="input/metadata.xlsx")
    parser.add_argument("--infer", action="store_true",
                        help="Run without Excel: infer Q/S numbers from filenames (best-effort)")
    parser.add_argument("--output", default="output/renamed")
    args = parser.parse_args()

    if not os.path.exists(args.zip):
        print(f"ERROR: ZIP not found: {args.zip}")
        sys.exit(1)

    excel_path = args.excel if os.path.exists(args.excel) else None
    if excel_path is None and not args.infer:
        print(f"ERROR: Excel not found: {args.excel}\nIf you want to run without the Excel, re-run with --infer to use filename heuristics.")
        sys.exit(1)

    rename_questions(args.zip, excel_path, args.output)


if __name__ == "__main__":
    main()