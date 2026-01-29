def test_list_presets(client):
    """Test listing model presets."""
    response = client.get("/api/v1/models/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) == 3  # SIR, SEIR, SIS

    preset_names = [p["name"] for p in data["presets"]]
    assert "SIR" in preset_names
    assert "SEIR" in preset_names
    assert "SIS" in preset_names


def test_preset_has_required_fields(client):
    """Test that presets have all required fields."""
    response = client.get("/api/v1/models/presets")
    data = response.json()

    for preset in data["presets"]:
        assert "name" in preset
        assert "description" in preset
        assert "compartments" in preset
        assert "parameters" in preset
        assert "transitions" in preset
