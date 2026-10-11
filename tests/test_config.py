from src.utils.config import load_config


def test_default_config_loads():
    config = load_config("configs/phoenix.yaml")
    assert config.aoi_name == "greater_phoenix"
    assert config.hex_resolution == 8
