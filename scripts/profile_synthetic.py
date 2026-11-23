"""Print calibration metrics for the synthetic world (occupancy by band, volumes, rates).

Usage: .venv/bin/python scripts/profile_synthetic.py [as_of YYYY-MM-DD] [seed]
"""

from __future__ import annotations

import sys
import time as _time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from b4l import demand, synth, validate  # noqa: E402


def main() -> None:
    as_of = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 11, 23)
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261123
    t0 = _time.time()
    world = synth.build_world(seed, as_of)
    intents = demand.build_intents(world)
    t1 = _time.time()
    res = demand.simulate(world, intents)
    t2 = _time.time()
    print(f"intents={len(intents.df)} bookings={len(res.bookings)} waitlist={len(res.waitlist)} "
          f"build={t1 - t0:.1f}s simulate={t2 - t1:.1f}s")
    occ = validate.occupancy_by_band(world, res)
    print(occ.to_string())
    for name, value in validate.summary_stats(world, res).items():
        print(f"{name:40s} {value}")


if __name__ == "__main__":
    main()
