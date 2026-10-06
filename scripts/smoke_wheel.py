"""Check installed resources and canonical commands outside the checkout."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

import scicomap
from scicomap.datasets import load_hill_topography, load_pic, load_scan_image


def main() -> None:
    """Verify the wheel's data loaders, console entry point, and artifacts."""
    root = Path(__file__).resolve().parents[1]
    assert not Path(scicomap.__file__).resolve().is_relative_to(root)
    images = [load_hill_topography(), load_scan_image()]
    images.extend(load_pic(name) for name in ("grmhd", "vortex", "tng"))
    assert all(
        image.ndim == 2 and np.isfinite(image).all() for image in images
    )
    command = Path(sys.executable).with_name("scicomap")
    with TemporaryDirectory(prefix="scicomap-wheel-") as directory:
        cwd = Path(directory)
        plt.imsave(cwd / "input.png", np.arange(16).reshape(4, 4), cmap="gray")
        operations = [
            ["list"],
            ["check", "hawaii", "--type", "sequential"],
            ["doctor", "--out-dir", directory],
            ["preview", "hawaii", "--out", "preview.png"],
            [
                "compare",
                "hawaii",
                "viridis",
                "--image",
                "scan",
                "--out",
                "compare.png",
            ],
            [
                "fix",
                "hawaii",
                "--out",
                "fixed.png",
                "--export",
                "corrected.json",
            ],
            ["cvd", "hawaii", "--out", "cvd.png"],
            [
                "apply",
                "corrected.json",
                "--image",
                "input.png",
                "--out",
                "applied.png",
            ],
            ["wizard", "--cmap", "hawaii"],
            ["report", "--cmap", "hawaii", "--fix", "--out", "report"],
        ]
        for args in [["version"], *operations]:
            if args[0] != "version":
                args += ["--json"]
            result = subprocess.run(
                [str(command), *args],
                cwd=cwd,
                env={**os.environ, "MPLBACKEND": "Agg"},
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (args, result.stdout, result.stderr)
            if args[0] != "version":
                assert json.loads(result.stdout)["ok"], (args, result.stdout)
            else:
                assert scicomap.__version__ in result.stdout
        for name in ("preview", "compare", "fixed", "cvd", "applied"):
            assert (cwd / f"{name}.png").stat().st_size > 0
        assert (cwd / "report/report.json").exists()
        for path in (
            cwd / "corrected.json",
            cwd / "report/corrected-cmap.json",
        ):
            exported = json.loads(path.read_text(encoding="utf-8"))
            assert exported["scicomap_version"] == scicomap.__version__
            loaded = scicomap.SciCoMap(
                ctype=exported["family"], cmap=exported["rgba"]
            )
            expected = scicomap.ScicoSequential("hawaii")
            expected.unif_sym_cmap()
            np.testing.assert_array_equal(
                loaded.cmap(np.arange(loaded.cmap.N)),
                expected.cmap(np.arange(expected.cmap.N)),
            )
    print(
        "Wheel smoke checks passed: five data resources and 11 canonical commands."
    )


if __name__ == "__main__":
    main()
