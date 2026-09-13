import io
import sys
from contextlib import nullcontext

import pytest

from vloop.progress import Progress


class TerminalBuffer(io.StringIO):
    def isatty(self):
        return True


def test_phases_reuse_one_line_and_resume_counts(monkeypatch):
    output = TerminalBuffer()
    monkeypatch.setattr(sys, "stderr", output)
    with Progress("preparing") as progress:
        progress.update()
        progress.phase("labeling", total=5, initial=3, failed=1)
        assert progress.bar.initial == progress.bar.n == 3
        progress.status("loading model")
        progress.status("labeling")
        for _ in range(2):
            progress.update(failed=1)
        assert progress.bar.n == 5
    assert "labeling" in output.getvalue()
    assert "failed=1" in output.getvalue()
    assert "100%" in output.getvalue()
    assert output.getvalue().count("\n") == 1


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_interrupted_item_is_not_counted_and_bar_closes(monkeypatch, failure):
    output = TerminalBuffer()
    monkeypatch.setattr(sys, "stderr", output)
    visited = []

    def inputs():
        for index in range(10):
            visited.append(index)
            yield index

    with pytest.raises(failure), Progress("checking", total=10) as progress:
        for index in progress.track(inputs()):
            if index == 2:
                raise failure()
            continue
    assert visited == [0, 1, 2]  # No eager consumption of the input iterator
    assert progress.bar.n == 2
    assert progress.bar.disable
    assert output.getvalue().count("\n") == 1
    assert "100%" not in output.getvalue()


@pytest.mark.parametrize("total", [None, 0])
def test_unknown_and_empty_phases_do_not_invent_completion(monkeypatch, total):
    output = TerminalBuffer()
    monkeypatch.setattr(sys, "stderr", output)
    with Progress("checking", total=total) as progress:
        assert list(progress.track(iter(()))) == []
    assert progress.bar.n == 0
    assert "100%" not in output.getvalue()


@pytest.mark.parametrize("failure", [None, KeyboardInterrupt])
def test_redirected_output_has_no_progress_and_keeps_errors(monkeypatch, failure):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stderr", output)
    context = pytest.raises(failure) if failure else nullcontext()
    with context, Progress("first", total=2) as progress:
        progress.update()
        progress.phase("second", total=1)
        print("error detail", file=sys.stderr)
        if failure:
            raise failure()
        progress.update()
    assert output.getvalue() == "error detail\n"
