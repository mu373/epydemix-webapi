"""End-to-end vaccination tests against custom models.

V-SEIHR is exercised separately in ``tests/test_simulations_v_seihr.py``.
Tests here verify validation paths and source/target resolution on a custom
2- or 3-compartment model with vaccinated twin compartments, with no
preset-default dependence.
"""


def _custom_vax_model_request(**overrides):
    """Minimal custom S/S_vax/I model with vaccination wired explicitly."""
    request = {
        "model": {
            "compartments": ["S", "S_vax", "I"],
            "parameters": {
                "transmission_rate": 0.3,
                "recovery_rate": 0.0,
            },
            "transitions": [
                {
                    "source": "S",
                    "target": "I",
                    "kind": "mediated",
                    "params": ["transmission_rate", "I"],
                },
                {
                    "source": "S_vax",
                    "target": "I",
                    "kind": "mediated",
                    "params": ["transmission_rate", "I"],
                },
            ],
        },
        "population": {
            "source": "custom",
            "name": "tiny",
            "age_groups": {"all": 1_000_000},
            "contact_matrices": {"all": [[1.0]]},
        },
        "simulation": {
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
            "Nsim": 5,
            "seed": 7,
        },
        "initial_conditions": {
            "method": "absolute",
            "compartments": {
                "S": [1_000_000],
                "S_vax": [0],
                "I": [0],
            },
        },
        "vaccination": {
            "source_compartment": "S",
            "target_compartment": "S_vax",
            "campaigns": [
                {
                    "start_date": "2025-01-05",
                    "end_date": "2025-01-15",
                    "rollout": {"type": "flat_count", "daily_doses": 10_000},
                }
            ],
        },
    }
    for key, value in overrides.items():
        request[key] = value
    return request


def test_vaccination_custom_model_runs(client):
    response = client.post("/api/v1/simulations", json=_custom_vax_model_request())
    assert response.status_code == 200, response.text
    data = response.json()
    assert "S_to_S_vax" in data["results"]["transitions"]["data"]


def test_vaccination_missing_source_target_on_custom_model(client):
    request = _custom_vax_model_request()
    request["vaccination"]["source_compartment"] = None # Missing source
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "source_compartment" in detail


def test_vaccination_invalid_source_compartment(client):
    request = _custom_vax_model_request()
    request["vaccination"]["source_compartment"] = "ghost" # Nonexistent compartment
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "ghost" in detail
    assert "source_compartment" in detail


def test_vaccination_source_equals_target(client):
    request = _custom_vax_model_request()
    request["vaccination"]["target_compartment"] = "S" # S -> S is not allowed
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "distinct" in detail


def test_vaccination_invalid_age_group(client):
    request = _custom_vax_model_request()
    request["vaccination"]["campaigns"][0]["target_age_groups"] = ["nonexistent"] # Target age group does not exist in population
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    assert "nonexistent" in response.json()["detail"]


def test_vaccination_duplicate_target_age_groups(client):
    request = _custom_vax_model_request()
    request["vaccination"]["campaigns"][0]["target_age_groups"] = ["all", "all"]
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    # Pydantic validators raise during request parsing, so the detail is the
    # standard list-of-errors shape rather than a plain string.
    detail_str = str(response.json()["detail"])
    assert "unique" in detail_str


def test_vaccination_invalid_window(client):
    request = _custom_vax_model_request()
    request["vaccination"]["campaigns"][0]["end_date"] = "2024-01-01" # End date before simulation start date
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422


def test_vaccination_empty_campaigns(client):
    request = _custom_vax_model_request()
    request["vaccination"]["campaigns"] = []
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    assert "campaigns" in response.json()["detail"]


def test_vaccination_window_outside_simulation(client):
    """Campaign window outside the simulation window should run cleanly with zero doses."""
    request = _custom_vax_model_request()
    request["vaccination"]["campaigns"][0]["start_date"] = "2030-01-01"
    request["vaccination"]["campaigns"][0]["end_date"] = "2030-01-31"
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text


def test_vaccination_invalid_rollout_type(client):
    """Unknown rollout discriminator should 422 via the Pydantic union."""
    request = _custom_vax_model_request()
    request["vaccination"]["campaigns"][0]["rollout"] = {
        "type": "ramp_count",
        "peak_daily_doses": 1000,
    }
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422


def test_vaccination_metadata_echoed(client):
    response = client.post("/api/v1/simulations", json=_custom_vax_model_request())
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["vaccination"]["source_compartment"] == "S"
    assert metadata["vaccination"]["target_compartment"] == "S_vax"
    assert len(metadata["vaccination"]["campaigns"]) == 1


def test_vaccination_dose_count_close_to_expected(client):
    """Cumulative S to S_vax transitions should be close to daily_doses * days."""
    request = _custom_vax_model_request()
    request["simulation"]["Nsim"] = 20
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    data = response.json()
    transitions = data["results"]["transitions"]["data"]["S_to_S_vax"]
    # Find a quantile series (median) over the campaign window.
    quantile_data = next(iter(transitions.values()))
    median_key = "median" if "median" in quantile_data else "0.5"
    series = quantile_data[median_key]
    dates = data["results"]["transitions"]["dates"]
    in_window = [v for d, v in zip(dates, series) if "2025-01-05" <= d <= "2025-01-15"]
    # 11 days, 10000 daily doses, dt=1.0; tolerate 30% noise across the median.
    expected = 10_000 * 11
    cumulative = sum(in_window)
    assert 0.5 * expected < cumulative < 1.5 * expected, (
        f"cumulative S_to_S_vax = {cumulative}, expected ~{expected}"
    )
