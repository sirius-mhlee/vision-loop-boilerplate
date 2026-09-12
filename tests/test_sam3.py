from dataclasses import replace

import pytest

from vloop import sam3


@pytest.mark.parametrize("device,index", [("cuda", 0), ("cuda:0", 0), ("cuda:1", 1)])
def test_cuda_alias_is_resolved_before_device_initialization(project, monkeypatch, device, index):
    torch = pytest.importorskip("torch")
    selected = []
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda device: selected.append(torch.cuda._get_device_index(device)),
    )
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)

    def stop_before_model_loading(cfg):
        raise RuntimeError("reached model loading")

    monkeypatch.setattr(sam3, "configure_fiftyone", stop_before_model_loading)
    with pytest.raises(RuntimeError, match="reached model loading"):
        with sam3.Sam3Labeler(replace(project, device=device)):
            pytest.fail("Model loading was not stopped")
    assert selected == [index]
