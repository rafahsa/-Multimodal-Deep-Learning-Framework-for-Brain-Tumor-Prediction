#!/usr/bin/env python3
"""Quick verification that selected patients exist and can be loaded."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / 'data' / 'processed' / 'stage_4_resize' / 'train'
PATIENTS_FILE = PROJECT_ROOT / 'data' / 'selected_patients.txt'

patients = [p.strip() for p in open(PATIENTS_FILE).readlines() if p.strip()]
print(f"Total patients: {len(patients)}")

missing = []
for p in patients:
    found = False
    for c in ['LGG', 'HGG']:
        if (DATA_ROOT / c / p).exists():
            found = True
            break
    if not found:
        missing.append(p)

if missing:
    print(f"Missing: {len(missing)}")
    print(missing)
else:
    print("✓ All patients found in data directory")

