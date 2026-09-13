import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vloop.fiftyone import close_session


@pytest.mark.parametrize("owned", [False, True])
def test_session_shutdown_only_arms_timer_for_owned_server(monkeypatch, owned):
    import vloop.fiftyone as module

    child, worker = Mock(), Mock()
    child.children.return_value = [worker]
    services = {5151: SimpleNamespace(child=child)} if owned else {}
    monkeypatch.setitem(
        sys.modules,
        "fiftyone.core.session",
        SimpleNamespace(session=SimpleNamespace(_server_services=services)),
    )
    # All processes are doubles; this unit test does not need psutil installed.
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(NoSuchProcess=ProcessLookupError))
    timer = Mock()
    factory = Mock(return_value=timer)
    monkeypatch.setattr(module.threading, "Timer", factory)
    session = Mock(server_port=5151)
    close_session(session)
    session.close.assert_called_once_with()
    if owned:
        timer.start.assert_called_once_with()
        timer.cancel.assert_called_once_with()
        delay, force_close = factory.call_args.args
        assert delay == 5
        child.kill.assert_not_called()
        worker.kill.assert_not_called()
        worker.kill.side_effect = ProcessLookupError  # Worker already exited.
        force_close()  # Simulate only this server exceeding its shutdown deadline.
        child.kill.assert_called_once_with()
        worker.kill.assert_called_once_with()
    else:
        factory.assert_not_called()
        child.kill.assert_not_called()


def test_review_interrupt_uses_bounded_shutdown_and_cli_exit_code(project, monkeypatch):
    import vloop.cli as cli
    import vloop.review as module

    session = Mock()
    monkeypatch.setattr(
        module, "configure_fiftyone", lambda cfg: Mock(launch_app=lambda *a, **k: session)
    )
    monkeypatch.setattr(module, "load_review_dataset", lambda cfg: Mock())

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(module.time, "sleep", interrupt)
    close = Mock()
    monkeypatch.setattr(module, "close_session", close)
    monkeypatch.setattr(
        module,
        "prepare_review",
        lambda *a, **k: dict(
            job_id="review-test",
            status="completed",
            dataset_version=None,
            result_dir="test",
            initialized=0,
            preserved=0,
            unavailable=0,
            source_job_id=None,
            invalidated=0,
        ),
    )
    assert cli.main(["review", "--config", str(project.config_path), "--no-browser"]) == 130
    close.assert_called_once_with(session)
