import json

from famous_vits.cli import main


def test_cli_lists_models(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "vit" in output
    assert "maxvit_tiny" in output


def test_cli_validates_yaml_config(capsys):
    assert main(["validate-config", "--config", "configs/smoke_test.yaml"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "train"
    assert payload["runtime"]["smoke_test"] is True


def test_cli_smoke_train_and_infer(tmp_path, capsys):
    checkpoint = tmp_path / "vit.pt"
    kwargs = json.dumps({"embed_dim": 24, "depth": 1, "num_heads": 3})
    assert (
        main(
            [
                "train",
                "--model",
                "vit",
                "--num-classes",
                "2",
                "--in-chans",
                "1",
                "--batch-size",
                "2",
                "--epochs",
                "1",
                "--smoke-test",
                "--model-kwargs",
                kwargs,
                "--output",
                str(checkpoint),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["infer", "--checkpoint", str(checkpoint), "--smoke-test"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["input"] == "synthetic"
    assert len(payload[0]["predictions"]) == 2
