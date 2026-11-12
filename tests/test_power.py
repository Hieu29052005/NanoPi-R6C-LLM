from conftest import load_script

power = load_script("power_monitor")


def test_parse_single_power_value():
    sample = power.parse_power_line("6.75")
    assert sample is not None
    assert sample.power_w == 6.75


def test_parse_voltage_current_pair():
    sample = power.parse_power_line("12.0, 0.5")
    assert sample.voltage_v == 12.0
    assert sample.current_a == 0.5
    assert sample.power_w == 6.0


def test_parse_key_value_and_json():
    kv = power.parse_power_line("V=5.0 I=2.0")
    assert kv.voltage_v == 5.0
    assert kv.current_a == 2.0
    assert kv.power_w == 10.0

    js = power.parse_power_line('{"voltage": 9.0, "current": 1.5}')
    assert js.power_w == 13.5


def test_parse_empty_and_garbage():
    assert power.parse_power_line("") is None
    assert power.parse_power_line("no numbers here") is None


def test_synthetic_monitor_summary(tmp_path):
    meter = power.SyntheticMeter(base_w=5.0, noise_w=0.0, voltage=10.0)
    out = tmp_path / "power.csv"
    summary = power.run_monitor(
        meter, duration_s=0, interval_s=0, out_path=str(out), max_samples=3, printer=None
    )

    assert summary["samples"] == 3
    assert summary["power_w"]["mean"] == 5.0
    assert summary["energy_wh"] >= 0
    assert out.exists()
    assert "power_w" in out.read_text(encoding="utf-8")
