"""End-to-end tests for tsearch/__main__.py — full serial pipeline.

Tests run the complete main() pipeline in serial mode (executorlib=False),
verify output formats (CSVs, trajectories, metadata), and test resume/continuation.
"""

import pytest
import os
import csv
import shutil
import glob
from pathlib import Path
from ase.io import read, write, Trajectory

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Common FAIRChem config block (reused across tests)
_FAIRCHEM_BLOCK = """\
[FAIRChemCalculator]
device = cuda
name_or_path = uma-s-1p1
task_name = oc20

[MDMin]
dt = 0.05
maxstep = 0.1
"""


def _write_config(tmp_path, content):
    (tmp_path / "config.ini").write_text(content)


@pytest.mark.gpu
@pytest.mark.slow
class TestMainPipeline:
    """End-to-end tests running the full main() in serial mode."""

    def test_serial_minimization_pipeline(self, tmp_path, monkeypatch):
        """Run Minimization end-to-end: config -> main() -> verify outputs."""
        monkeypatch.chdir(tmp_path)

        traj_dir = tmp_path / "trajes"
        traj_dir.mkdir()
        atoms = read(str(FIXTURES_DIR / "minimization_input.traj"))
        write(str(traj_dir / "input.traj"), atoms)

        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = Minimization
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 50
zip = False
{_FAIRCHEM_BLOCK}
""")

        from tsearch.__main__ import main
        main()

        # Verify directory structure
        assert os.path.exists("traj_files_ordered.json")
        assert os.path.isdir("Minimization_status_csvs")
        assert os.path.isdir("Minimization_trajes")
        assert os.path.isdir("Minimization_debug_zips")

        # Verify CSV format: job_id,rank,status
        csv_files = glob.glob("Minimization_status_csvs/*.csv")
        assert len(csv_files) >= 1
        with open(csv_files[0], 'r') as f:
            for parts in csv.reader(f):
                assert len(parts) == 3, f"Minimization CSV should have 3 columns, got: {parts}"
                assert parts[2] in ("converged", "not_converged") or parts[2].startswith("error")

        # Verify output trajectory
        traj_files = glob.glob("Minimization_trajes/*.traj")
        assert len(traj_files) >= 1
        result = read(traj_files[0])
        assert "converged" in result.info
        assert "src_index" in result.info
        assert result.info["converged"] in (0, 1)

    def test_serial_neb_pipeline_output_format(self, tmp_path, monkeypatch):
        """Run NEB end-to-end and thoroughly validate output format."""
        monkeypatch.chdir(tmp_path)

        traj_dir = tmp_path / "trajes"
        traj_dir.mkdir()
        shutil.copy(str(FIXTURES_DIR / "oc_neb_pair.traj"), str(traj_dir / "neb_input.traj"))

        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = NEB
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 30
zip = False
{_FAIRCHEM_BLOCK}

[ourNEB]
relax_endpoints = False
interpolate_method = False
num_frames = 10
batch_size = 4
DNEB = False
intermediate_minima_check_step = 0

[BaseNEB]
k = 5
climb = True
method = improvedtangent
allow_shared_calculator = True
""")

        from tsearch.__main__ import main
        main()

        # Verify CSV format: job_id,rank,sub_band_id,status
        csv_files = glob.glob("NEB_status_csvs/*.csv")
        assert len(csv_files) >= 1
        with open(csv_files[0], 'r') as f:
            for parts in csv.reader(f):
                assert len(parts) == 4, f"NEB CSV should have 4 columns, got: {parts}"
                status = parts[3]
                assert status in ("converged", "converged_CI", "not_converged") or status.startswith("error")

        # Validate all NEB output trajectory metadata
        traj_files = glob.glob("NEB_trajes/*.traj")
        assert len(traj_files) >= 1
        results = read(traj_files[0], index=":")
        assert len(results) == 10, f"Expected 10 band images, got {len(results)}"

        required_keys = [
            "src_index", "image_idx", "subband_idx", "image_type",
            "effective_fmax", "image_converged", "band_converged",
            "band_converged_CI", "nimages",
        ]
        valid_types = {"endpoint", "intermediate_minimum", "climbing", "regular"}
        image_indices = []
        has_climbing = False
        for img in results:
            for key in required_keys:
                assert key in img.info, f"Missing '{key}' in image {img.info.get('image_idx')}"
            assert img.info["image_type"] in valid_types
            assert img.info["src_index"] == 0
            assert isinstance(img.info["effective_fmax"], float)
            assert isinstance(img.info["image_converged"], bool)
            assert isinstance(img.info["band_converged"], bool)
            image_indices.append(img.info["image_idx"])
            if img.info["image_type"] == "climbing":
                has_climbing = True
                # CI images must have eigenmode, barrier, dE
                assert "eigenmode" in img.info, "CI image missing eigenmode"
                assert "barrier" in img.info, "CI image missing barrier"
                assert "dE" in img.info, "CI image missing dE"

        # Endpoints at positions 0 and N-1
        assert results[0].info["image_type"] == "endpoint"
        assert results[-1].info["image_type"] == "endpoint"
        # Should have at least one climbing image (climb=True)
        assert has_climbing, "No climbing image found with climb=True"
        # image_idx should be 0 through N-1
        assert sorted(image_indices) == list(range(10))

        # Band plot should exist
        assert os.path.exists("diffusion_barrier_0.png"), "Band plot not created"

    def test_serial_dimer_pipeline_output_format(self, tmp_path, monkeypatch):
        """Run Dimer end-to-end and thoroughly validate output format."""
        monkeypatch.chdir(tmp_path)

        traj_dir = tmp_path / "trajes"
        traj_dir.mkdir()
        shutil.copy(str(FIXTURES_DIR / "bulk_crystal.traj"), str(traj_dir / "bulk.traj"))

        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = Dimer
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 30
zip = False
{_FAIRCHEM_BLOCK}

[ourDimer]
dataset_type = bulk
reaction_types = vacancy
num_attempts_per_type = 1
supercell = True

[DimerControl]
initial_eigenmode_method = displacement
maximum_translation = 0.1
dimer_separation = 0.01
""")

        from tsearch.__main__ import main
        main()

        # Verify CSV format: job_id,rank,attempt_id,selected_index,status
        csv_files = glob.glob("Dimer_status_csvs/*.csv")
        assert len(csv_files) >= 1
        with open(csv_files[0], 'r') as f:
            for parts in csv.reader(f):
                assert len(parts) == 5, f"Dimer CSV should have 5 columns, got: {parts}"
                status = parts[4]
                valid_statuses = {
                    "converged", "converged_after_extension",
                    "converged_to_desorption",
                    "not_converged", "not_converged_after_extension",
                    "not_converged_StopRun",
                }
                assert status in valid_statuses or status.startswith("error"), \
                    f"Unexpected Dimer status: {status}"

        # Validate Dimer output trajectory metadata
        traj_files = glob.glob("Dimer_trajes/*.traj")
        assert len(traj_files) >= 1
        result = read(traj_files[0])
        required_keys = [
            "src_index", "attempt_id", "reaction_type", "eigenmode",
            "converged", "stoprun", "selected_index", "curvature",
        ]
        for key in required_keys:
            assert key in result.info, f"Dimer output missing '{key}'"
        assert result.info["reaction_type"] == "vacancy"
        assert result.info["converged"] in (0, 1)
        assert result.info["stoprun"] in (0, 1)
        # eigenmode should be an array with shape (natoms, 3)
        import numpy as np
        eigenmode = np.array(result.info["eigenmode"])
        assert eigenmode.shape == (len(result), 3)

    def test_neb_resume_continuation(self, tmp_path, monkeypatch):
        """Run NEB with few steps (not_converged), then resume with more steps.

        Validates: archive created, continuation picks up from previous band,
        new output overwrites old entries.
        """
        monkeypatch.chdir(tmp_path)

        traj_dir = tmp_path / "trajes"
        traj_dir.mkdir()
        shutil.copy(str(FIXTURES_DIR / "oc_neb_pair.traj"), str(traj_dir / "neb_input.traj"))

        # First run: tight fmax + very few steps -> not_converged
        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = NEB
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.001
steps = 5
zip = False
{_FAIRCHEM_BLOCK}

[ourNEB]
relax_endpoints = False
interpolate_method = False
num_frames = 10
batch_size = 4
DNEB = False
intermediate_minima_check_step = 0

[BaseNEB]
k = 5
climb = True
method = improvedtangent
allow_shared_calculator = True
""")

        from tsearch.__main__ import main
        main()

        # Verify first run produced not_converged
        csv_files = glob.glob("NEB_status_csvs/*.csv")
        assert len(csv_files) >= 1
        with open(csv_files[0], 'r') as f:
            first_status = f.read().strip()
        assert "not_converged" in first_status, f"Expected not_converged, got: {first_status}"

        # Count first run output frames
        traj_files = glob.glob("NEB_trajes/*.traj")
        first_results = read(traj_files[0], index=":")
        first_nimages = len(first_results)
        assert first_nimages == 10

        # Second run: resume not_converged with more steps
        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = NEB
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 30
zip = False
run_jobs = not_converged
{_FAIRCHEM_BLOCK}

[ourNEB]
relax_endpoints = False
interpolate_method = False
num_frames = 10
batch_size = 4
DNEB = False
intermediate_minima_check_step = 0

[BaseNEB]
k = 5
climb = True
method = improvedtangent
allow_shared_calculator = True
""")

        main()

        # Archive should exist
        archives = glob.glob("NEB_status_csvs/previous_*.zip")
        assert len(archives) >= 1, "CSV archive not created on NEB resume"
        traj_archives = glob.glob("NEB_trajes/previous_*.zip")
        assert len(traj_archives) >= 1, "Traj archive not created on NEB resume"

        # New output should have 10 images (the resumed band)
        traj_files = glob.glob("NEB_trajes/*.traj")
        resumed_results = read(traj_files[0], index=":")
        assert len(resumed_results) == 10, f"Expected 10 images after resume, got {len(resumed_results)}"

        # All images should have proper metadata
        for img in resumed_results:
            assert "src_index" in img.info
            assert "image_idx" in img.info
            assert "image_type" in img.info

    def test_dimer_resume_continuation(self, tmp_path, monkeypatch):
        """Run Dimer with few steps (not_converged), then resume with more steps.

        Validates: archive created, continuation picks up, output format correct.
        """
        monkeypatch.chdir(tmp_path)

        traj_dir = tmp_path / "trajes"
        traj_dir.mkdir()
        shutil.copy(str(FIXTURES_DIR / "oc_adsorbate_slab.traj"), str(traj_dir / "slab.traj"))

        # First run: tight fmax + few steps -> not_converged
        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = Dimer
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.001
steps = 5
zip = False
{_FAIRCHEM_BLOCK}

[ourDimer]
dataset_type = oc
reaction_types = adsorbate_atom
num_attempts_per_type = 1
supercell = False

[DimerControl]
initial_eigenmode_method = displacement
maximum_translation = 0.1
dimer_separation = 0.01
""")

        from tsearch.__main__ import main
        main()

        # First run should produce output
        csv_files = glob.glob("Dimer_status_csvs/*.csv")
        assert len(csv_files) >= 1
        traj_files = glob.glob("Dimer_trajes/*.traj")
        assert len(traj_files) >= 1
        first_result = read(traj_files[0])
        assert "attempt_id" in first_result.info

        # Second run: resume not_converged
        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = Dimer
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 30
zip = False
run_jobs = not_converged
{_FAIRCHEM_BLOCK}

[ourDimer]
dataset_type = oc
reaction_types = adsorbate_atom
num_attempts_per_type = 1
supercell = False

[DimerControl]
initial_eigenmode_method = displacement
maximum_translation = 0.1
dimer_separation = 0.01
""")

        main()

        # Archive should exist
        archives = glob.glob("Dimer_status_csvs/previous_*.zip")
        assert len(archives) >= 1, "CSV archive not created on Dimer resume"

        # New output should have the resumed attempt
        traj_files = glob.glob("Dimer_trajes/*.traj")
        resumed_result = read(traj_files[0])
        assert "src_index" in resumed_result.info
        assert "attempt_id" in resumed_result.info
        assert "eigenmode" in resumed_result.info

    def test_minimization_resume(self, tmp_path, monkeypatch):
        """Run Minimization with few steps, then resume with run_jobs=not_converged."""
        monkeypatch.chdir(tmp_path)

        traj_dir = tmp_path / "trajes"
        traj_dir.mkdir()
        atoms = read(str(FIXTURES_DIR / "minimization_input.traj"))
        write(str(traj_dir / "input.traj"), atoms)

        # First run: very few steps, likely not converged
        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = Minimization
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.001
steps = 5
zip = False
{_FAIRCHEM_BLOCK}
""")

        from tsearch.__main__ import main
        main()

        assert os.path.exists("traj_files_ordered.json")
        csv_files = glob.glob("Minimization_status_csvs/*.csv")
        assert len(csv_files) >= 1

        # Second run: resume not_converged with more steps
        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = Minimization
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 50
zip = False
run_jobs = not_converged
{_FAIRCHEM_BLOCK}
""")

        main()

        # Verify archive was created
        archives = glob.glob("Minimization_status_csvs/previous_*.zip")
        assert len(archives) >= 1, "Archive should have been created on resume"

        # Output traj should have the resumed result
        traj_files = glob.glob("Minimization_trajes/*.traj")
        result = read(traj_files[0])
        assert "converged" in result.info
        assert "src_index" in result.info
