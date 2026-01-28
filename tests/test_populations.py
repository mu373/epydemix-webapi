def test_list_populations(client):
    """Test listing available populations."""
    response = client.get("/api/v1/populations")
    assert response.status_code == 200

    data = response.json()
    assert "populations" in data
    assert "total" in data
    assert data["total"] > 0

    # Check that United_States is in the list
    names = [p["name"] for p in data["populations"]]
    assert "United_States" in names


def test_get_population_detail(client):
    """Test getting population details."""
    response = client.get("/api/v1/populations/United_States")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "United_States"
    assert "total_population" in data
    assert data["total_population"] > 0
    assert "age_groups" in data
    assert len(data["age_groups"]) > 0
    assert "available_layers" in data


def test_get_population_contacts(client):
    """Test getting contact matrices for a population."""
    response = client.get("/api/v1/populations/United_States/contacts")
    assert response.status_code == 200

    data = response.json()
    assert data["population_name"] == "United_States"
    assert "layers" in data
    assert "age_groups" in data

    # Should have contact matrices for standard layers
    assert len(data["layers"]) > 0
    assert "overall" in data


def test_get_population_contacts_with_layers_filter(client):
    """Test getting specific contact layers."""
    response = client.get(
        "/api/v1/populations/United_States/contacts",
        params={"layers": ["home", "work"]},
    )
    assert response.status_code == 200

    data = response.json()
    assert "home" in data["layers"]
    assert "work" in data["layers"]


def test_get_nonexistent_population(client):
    """Test getting a population that doesn't exist."""
    response = client.get("/api/v1/populations/Nonexistent_Country")
    assert response.status_code in [404, 500]  # Either not found or internal error
