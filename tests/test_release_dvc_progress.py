import io
import logging
import sys
from contextlib import nullcontext
from dataclasses import replace

import pytest

from vloop.release_dvc import LABEL_TARGETS, _dvc_progress, add_and_push


class TerminalBuffer(io.StringIO):
    @property
    def encoding(self):
        return "utf-8"

    def isatty(self):
        return True


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_progress_hides_dvc_bars_and_restores_logging(monkeypatch, caplog, failure):
    dvc = pytest.importorskip("dvc.progress")
    data = pytest.importorskip("dvc_data.callbacks")
    from dvc.ui import ui
    from rich.console import Console

    monkeypatch.setenv("TERM", "xterm-256color")
    stream = TerminalBuffer()
    monkeypatch.setattr(sys, "stderr", stream)
    console = Console(file=stream, force_terminal=True, force_interactive=True)
    monkeypatch.setattr(ui, "error_console", console)
    loggers = [logging.getLogger(name) for name in ("dvc.progress", "dvc_data.callbacks")]
    levels = [logger.level for logger in loggers]
    context = pytest.raises(failure) if failure else nullcontext()

    with context, _dvc_progress("vloop test: DVC", 2) as bar:
        for cls in (dvc.Tqdm, data.Tqdm):
            with cls(total=1, desc="nested DVC bar", leave=True) as nested:
                nested.update(1)
        with ui.status("DVC status spinner") as status:
            status.update("Checking graph")
        console.print("DVC console message remains visible")
        logging.getLogger("dvc.repo").warning("DVC warning remains visible")
        logging.getLogger("dvc_data.index").error("DVC error remains visible")
        bar.update(1)
        if failure:
            raise failure()
        bar.update(1)

    output = stream.getvalue()
    assert "vloop test: DVC" in output
    assert "nested DVC bar" not in output
    assert "Checking graph" not in output
    assert "DVC console message remains visible" in output
    assert ("100%" in output) == (failure is None)
    assert "DVC warning remains visible" in caplog.text
    assert "DVC error remains visible" in caplog.text
    assert [logger.level for logger in loggers] == levels
    assert console.is_interactive is True

    # A later training/evaluation bar must still render after success or interruption.
    with dvc.tqdm(total=1, desc="model progress") as bar:
        bar.update(1)
    assert "model progress" in stream.getvalue()


def test_progress_does_not_fill_redirected_logs(monkeypatch):
    pytest.importorskip("dvc")
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    with _dvc_progress("vloop test: DVC", 100) as bar:
        for _ in range(100):
            bar.update(1)
    assert stream.getvalue() == ""


def test_release_progress_stays_on_one_terminal_line(project, tmp_path, monkeypatch):
    pytest.importorskip("dvc")
    from dvc.ui import ui
    from rich.console import Console

    monkeypatch.setenv("TERM", "xterm-256color")
    stream = TerminalBuffer()
    monkeypatch.setattr(sys, "stderr", stream)
    console = Console(file=stream, force_terminal=True, force_interactive=True, width=80)
    monkeypatch.setattr(ui, "error_console", console)
    cfg = replace(project, dvc_remote=tmp_path / "remote")
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)
    work = tmp_path / "work"
    for target in [*LABEL_TARGETS, "dataset/images/ab", "dataset/images/cd"]:
        folder = work / target
        folder.mkdir(parents=True)
        (folder / "example.txt").write_text("progress regression fixture")

    # Run real add/push: each add opens a Rich status that used to disrupt the bar.
    add_and_push(cfg, work)

    output = stream.getvalue()
    assert "vloop release: DVC" in output and "100%" in output
    assert output.count("\r") > 1  # Multiple in-place refreshes
    assert output.count("\n") == 1  # Only the final completed line
    assert "Checking graph" not in output
    assert console.is_interactive is True
