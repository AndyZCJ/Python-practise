from parser import parse_line


def test_valid_line():

    line = "2026-01-26 10:00:00 INFO hello"

    record = parse_line(line)

    # ===== TODO =====
    assert record is not None

    assert record["level"] == "INFO"

    assert record["message"] == "hello"

    assert "timestamp" in record