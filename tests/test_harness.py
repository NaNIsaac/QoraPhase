"""benchmarks/harness.py: one tiny case runs, and the JSON lines it writes compare."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import harness


def test_a_case_reports_compile_time_and_every_timed_run():
    r = harness.run_case("ER", (16, 16), iterations=3, warmup=1, repeat=4)
    assert r.shape == "16x16" and r.iterations == 3
    assert r.compile_s > 0 and len(r.times_s) == 4 and all(t > 0 for t in r.times_s)
    q1, q3 = r.quartiles()
    assert min(r.times_s) <= q1 <= r.median <= q3 <= max(r.times_s)


def test_compare_reads_the_json_lines_back(tmp_path, capsys):
    r = harness.run_case("RAAR", (16, 16), iterations=2, warmup=1, repeat=2)
    before, after = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    before.write_text(json.dumps(asdict(r)) + "\n", encoding="utf-8")
    slower = harness.Result(**{**asdict(r), "times_s": [2 * t for t in r.times_s]})
    after.write_text(json.dumps(asdict(slower)) + "\n", encoding="utf-8")
    harness.compare(str(before), str(after))
    assert "| RAAR | 16x16 |" in capsys.readouterr().out
    assert harness.load(str(after))[("RAAR", "16x16")].median == 2 * r.median
