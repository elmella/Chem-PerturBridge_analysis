#!/usr/bin/env python3
"""Build reviewer Table 7 without executing its notebook."""

from replicate_reviewer_tables import main


if __name__ == "__main__":
    raise SystemExit(main(fixed_table=7))
