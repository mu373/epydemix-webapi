from app.presets import PRESETS


def test_list_presets(client):
    """Listing returns one entry per registry definition."""
    response = client.get("/api/v1/models/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    response_names = [p["name"] for p in data["presets"]]
    assert set(response_names) == set(PRESETS)


def test_preset_has_required_fields(client):
    """Each preset entry carries the expected fields."""
    response = client.get("/api/v1/models/presets")
    data = response.json()

    for preset in data["presets"]:
        assert "name" in preset
        assert "description" in preset
        assert "compartments" in preset
        assert "parameters" in preset
        assert "transitions" in preset
