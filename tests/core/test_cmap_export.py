"""Reusable correction tables and their CLI consumers."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from typer.testing import CliRunner

import scicomap as sc
from scicomap import cli


@pytest.mark.parametrize("size", [7, 8])
def test_export_reloads_and_replays_ordered_corrections(
    tmp_path, size
) -> None:
    source = LinearSegmentedColormap.from_list(
        "source", [(0.1, 0.2, 0.3, 0.2), (0.9, 0.8, 0.7, 0.8)], N=size
    )
    chart = sc.ScicoSequential(source)
    chart.uniformize_cmap(lightness_rounding=np.float64(0))
    chart.symmetrize_cmap(bitonic=np.bool_(False), diffuse=False)
    chart.unif_sym_cmap(lightness_rounding=10, bitonic=False)
    exported_path = chart.export_cmap(tmp_path / "nested" / "map.json")
    assert exported_path.is_absolute()
    exported = json.loads(exported_path.read_text())
    assert exported["scicomap_version"] == sc.__version__
    assert exported["family"] == "sequential"
    assert exported["name"] == "source"
    reloaded = ListedColormap(exported["rgba"], name=exported["name"])
    assert reloaded.N == size
    np.testing.assert_array_equal(
        reloaded(np.arange(size)), chart.cmap(np.arange(size))
    )
    np.testing.assert_array_equal(
        reloaded(np.arange(size))[:, 3], source(np.arange(size))[:, 3]
    )
    np.testing.assert_array_equal(
        exported["source_rgba"], source(np.arange(size))
    )
    from_list = sc.SciCoMap(ctype=exported["family"], cmap=exported["rgba"])
    np.testing.assert_array_equal(
        from_list.cmap(np.arange(size)), exported["rgba"]
    )
    replayed = ListedColormap(exported["source_rgba"])
    for step in exported["transformations"]:
        parameters = {
            key: value for key, value in step.items() if key != "operation"
        }
        replayed = getattr(sc, step["operation"])(replayed, **parameters)
    np.testing.assert_array_equal(replayed(np.arange(size)), exported["rgba"])


def test_invalid_export_preserves_existing_file(tmp_path) -> None:
    chart = sc.SciCoMap(cmap=["black", "white"])
    with pytest.raises(ValueError):
        chart.uniformize_cmap(lightness_rounding=-1)
    path = chart.export_cmap(tmp_path / "map.json")
    assert json.loads(path.read_text())["transformations"] == []
    before = path.read_bytes()
    chart.cmap = ListedColormap([[np.nan, 0, 0, 1]])
    with pytest.raises(ValueError, match="finite"):
        chart.export_cmap(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("command", ["fix", "wizard"])
@pytest.mark.parametrize("as_json", [False, True])
def test_table_only_export_never_renders(
    tmp_path, monkeypatch, command, as_json
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Table-only export must not render or display")

    monkeypatch.setattr(sc.SciCoMap, "assess_cmap", forbidden)
    monkeypatch.setattr(plt, "show", forbidden)
    path = tmp_path / "map.json"
    args = [
        command,
        "--export",
        str(path),
        "--lightness-rounding",
        "0",
        "--no-bitonic",
    ]
    if command == "fix":
        args.append("hawaii")
    else:
        args += ["--cmap", "hawaii", "--fix", "--no-interactive"]
    if as_json:
        args.append("--json")
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    exported = json.loads(path.read_text())
    expected = sc.ScicoSequential("hawaii")
    expected.unif_sym_cmap(lightness_rounding=0, bitonic=False)
    np.testing.assert_array_equal(
        exported["rgba"], sc.cmath.get_ctab(expected.cmap)
    )
    assert exported["transformations"] == [
        {
            "operation": "unif_sym_cmap",
            "lightness_rounding": 0,
            "bitonic": False,
            "diffuse": True,
        }
    ]
    if as_json:
        assert json.loads(result.stdout)["data"]["artifacts"] == [
            {"kind": "color_table", "path": str(path), "map": "transformed"}
        ]


@pytest.mark.parametrize(
    "command",
    ["check", "preview", "compare", "fix", "cvd", "apply", "wizard", "report"],
)
def test_cli_commands_load_exported_tables(
    tmp_path, monkeypatch, command
) -> None:
    chart = sc.ScicoSequential("hawaii")
    chart.unif_sym_cmap(lightness_rounding=0)
    path = chart.export_cmap(tmp_path / "map.json")
    monkeypatch.setattr(sc.SciCoMap, "assess_cmap", lambda self: plt.figure())
    captured = []

    def capture(**kwargs):
        captured.extend(kwargs.get("cm_list", kwargs.get("cmap_list", [])))
        return plt.figure()

    monkeypatch.setattr(cli, "compare_cmap", capture)
    monkeypatch.setattr(cli, "plot_colorblind_vision", capture)
    args = [command, "--json"]
    if command in {"wizard", "report"}:
        args += ["--cmap", str(path)]
    else:
        args.append(str(path))
    if command == "compare":
        args.append("hawaii")
    if command not in {"check", "wizard"}:
        args += [
            "--out",
            str(tmp_path / ("report" if command == "report" else "out.png")),
        ]
    if command == "apply":
        image = tmp_path / "image.png"
        plt.imsave(image, np.array([[0, 1], [0.25, 0.75]]), cmap="gray")
        args += ["--image", str(image)]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    if command == "check":
        assert json.loads(result.stdout)["data"] == sc.diagnose_cmap(
            chart.cmap, chart.ctype
        )
    for used in captured[:1]:
        np.testing.assert_array_equal(
            used(np.arange(used.N)), sc.cmath.get_ctab(chart.cmap)
        )
    if command == "apply":
        np.testing.assert_allclose(
            plt.imread(tmp_path / "out.png"),
            cli._remap_image(image, chart.cmap, "luminance"),
            atol=1 / 255,
        )


@pytest.mark.parametrize(
    "content",
    [
        "{",
        "[]",
        '{"family": [], "name": "bad", "rgba": []}',
        '{"family": "sequential", "name": "bad", "rgba": [[2, 0, 0]]}',
    ],
)
def test_invalid_export_returns_json_without_artifacts(
    tmp_path, content
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content)
    result = CliRunner().invoke(
        cli.app,
        ["preview", str(path), "--out", str(tmp_path / "out.png"), "--json"],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["ok"] is False
    assert not (tmp_path / "out.png").exists()


def test_export_path_collisions_fail_before_writing(tmp_path) -> None:
    path = tmp_path / "out.json"
    result = CliRunner().invoke(
        cli.app, ["fix", "--out", str(path), "--export", str(path), "--json"]
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["errors"]
    assert not path.exists()
