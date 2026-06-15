#!/usr/bin/env python3
"""
Compatibility wrapper for Task 03.

Some users call the script with the underscore name `rename_questions.py`.
The canonical script in this repo is `renameQuestions.py` (camelCase). This
wrapper simply delegates to that module so both names work.
"""
import sys

try:
    from renameQuestions import main as _main
except Exception as exc:
    print(f"ERROR: Failed to import renameQuestions.py: {exc}")
    sys.exit(1)


if __name__ == "__main__":
    _main()
