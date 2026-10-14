import importlib.util
import json
from pathlib import Path

import numpy as np

from probebench.sequential import run


def test_analysis_retains_each_case_and_the_equal_acquisition_control(tmp_path):
    root = Path(__file__).parents[1]
    config = json.loads((root/"configs/measurement_schedule_development.json").read_text())
    config.update(replicates=1, exposure_budgets=[2,3], photons_per_exposure=[2000],
                  truths_opd_nm=config["truths_opd_nm"][:1], operating_reserve_bytes=16*1024**2)
    config["optics"]["pupil_pixels"] = 12
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path/"run"
    run(output,config_path)
    spec = importlib.util.spec_from_file_location("schedule_analysis",root/"scripts/analyze_measurement_schedule.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    analysis = module.analyze(output)
    assert analysis["episode_count"] == 6 and analysis["exposure_count"] == 15
    assert len(analysis["episodes"]) == 6 and len(analysis["statistics"]) == 6
    assert len(analysis["paired_comparisons"]) == 2
    assert analysis["paired_comparisons"][0]["fixed_minus_adaptive_wfe_nm"] == 0
    assert len(analysis["identical_fixed_adaptive_controls"]) == 1
    assert len(analysis["unprobed_likelihood_symmetry"]) == 2
    assert sum(r["fit_count"] for r in analysis["episodes"]) == 7
    assert sum(r["optimizer_starts"] for r in analysis["episodes"]) == 21
    # Compare against the separately derived runner table, including solver state.
    summary = json.loads((output/"summary.json").read_text())
    for saved, row in zip(summary,analysis["episodes"]):
        assert saved["episode_id"] == row["episode"]
        assert saved["residual_wfe_nm"] == row["final_wfe_nm"]
        assert row["wfe_modulo_conjugacy_nm"] <= row["final_wfe_nm"]
        assert saved["failed_fits"] == row["failed_fits"]
        assert saved["unconverged_fits"] == row["unconverged_fits"]
    assert all(np.isfinite(r["final_wfe_nm"]) for r in analysis["episodes"])
    assert "7 selected fits and 21 starts" in module.report(analysis)
