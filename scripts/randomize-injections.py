# Template-to-carrier assignment

import csv
import random
from collections import Counter
from pathlib import Path

def assign_templates(carriers, templates, n_per_template, seed):
    assert len(carriers) == len(templates) * n_per_template, (
        f"carrier count ({len(carriers)}) must equal "
        f"templates ({len(templates)}) * n_per_template ({n_per_template})"
    )

    assignment_pool = []
    for template_id in templates:
        assignment_pool.extend([template_id] * n_per_template)

    rng = random.Random(seed)
    rng.shuffle(assignment_pool)

    return dict(zip(carriers, assignment_pool))

def validate_assignment(assignment, expected_per_template, label):
    """Verify each template appears exactly expected_per_template times."""
    template_counts = Counter(assignment.values())
    for template_id, count in template_counts.items():
        assert count == expected_per_template, (
            f"[{label}] Template {template_id} used {count} times, "
            f"expected {expected_per_template}"
        )
    print(
        f"[{label}] Validated: {len(template_counts)} templates, "
        f"each used {expected_per_template} times"
    )

# 100 information-only synthetic carriers
info_carriers = list(range(1, 101))

# 10 attack templates per plausibility level, numbered 1-10 within each level
templates_per_level = list(range(1, 11))

SEEDS = {
    "low": 42,
    "medium": 43,
    "high": 44,
}

print("=" * 60)
print("CARRIER-INJECTION ASSIGNMENTS (synthetic info-only carriers)")
print("=" * 60)

assignments = {
    level: assign_templates(info_carriers, templates_per_level, 10, seed)
    for level, seed in SEEDS.items()
}

for level in ("low", "medium", "high"):
    validate_assignment(assignments[level], 10, level)

def write_csv(filename, assignments, carrier_ids):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "carrier_id", "corpus", "carrier_type",
            "low_template", "medium_template", "high_template",
        ])
        for cid in sorted(carrier_ids):
            writer.writerow([
                cid, "synthetic", "info",
                assignments["low"][cid],
                assignments["medium"][cid],
                assignments["high"][cid],
            ])

output_path = Path(__file__).resolve().parent / "template_assignments.csv"
write_csv(output_path, assignments, info_carriers)

print()
print("=" * 60)
print(f"Wrote {output_path}")
print("=" * 60)
print(f"Total rows: {len(info_carriers)} (synthetic info-only, with template assignments)")
