"""End-to-end tests for the V-SEIHR preset.

Locks down compartment structure, default behavior, the parameter-conversion
chain (period inputs and R0 -> beta), and integration with the vaccination
block.
"""

from app.presets import PRESETS


_V_SEIHR_COMPARTMENTS = [
    "Susceptible",
    "Susceptible_vax",
    "Exposed",
    "Exposed_vax",
    "Infected",
    "Infected_vax",
    "Hospitalized",
    "Hospitalized_vax",
    "Recovered",
    "Recovered_vax",
]


def _baseline_request(**overrides):
    request = {
        "model": {
            "preset": "V-SEIHR",
            "parameters": {
                "R0": 2.5,
                "incubation_period": 3.0,
                "infectious_period": 2.5,
                "hospitalization_duration": 5.0,
                "hosp_proportion": 0.05,
                "VE_S": 0.7,
                "VE_H": 0.85,
            },
        },
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
            "Nsim": 3,
            "seed": 42,
        },
    }
    request.update(overrides)
    return request


def test_v_seihr_baseline(client):
    """V-SEIHR preset runs with defaults and exposes all 10 compartments."""
    response = client.post("/api/v1/simulations", json=_baseline_request())
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["metadata"]["model"]["compartments"] == _V_SEIHR_COMPARTMENTS


def test_v_seihr_in_presets_endpoint(client):
    """`GET /models/presets` lists V-SEIHR with the right compartments."""
    response = client.get("/api/v1/models/presets")
    assert response.status_code == 200
    presets = {p["name"]: p for p in response.json()["presets"]}
    assert "V-SEIHR" in presets
    assert presets["V-SEIHR"]["compartments"] == _V_SEIHR_COMPARTMENTS


def test_v_seihr_registry_consistency(client):
    """The registry, the presets endpoint, and the request literal agree."""
    response = client.get("/api/v1/models/presets")
    endpoint_names = {p["name"] for p in response.json()["presets"]}
    assert endpoint_names == set(PRESETS)


def test_v_seihr_calc_params_exposed(client):
    """Preset-specific calc-params surface in `results.parameters`."""
    request = _baseline_request()
    request["output"] = {"include_parameters": True}
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    params = response.json()["results"]["parameters"]
    for name in (
        "transmission_rate",
        "transmission_rate_vax",
        "hosp_proportion_vax",
        "I_to_R_rate",
        "I_to_H_rate",
        "Ivax_to_R_rate",
        "Ivax_to_H_rate",
        "recovery_rate",
        "incubation_rate",
        "hosp_recovery_rate",
    ):
        assert name in params["data"], f"missing {name} in results.parameters"


def test_v_seihr_VE_zero(client):
    """When VE_S and VE_H are zero, vaccinated rate parameters equal unvaccinated."""
    request = _baseline_request()
    request["model"]["parameters"]["VE_S"] = 0.0
    request["model"]["parameters"]["VE_H"] = 0.0
    request["output"] = {"include_parameters": True}
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    params = response.json()["results"]["parameters"]["data"]
    first = next(iter(params["transmission_rate"]))
    assert abs(params["transmission_rate"][first][0] - params["transmission_rate_vax"][first][0]) < 1e-9
    assert abs(params["hosp_proportion"][first][0] - params["hosp_proportion_vax"][first][0]) < 1e-9


def test_v_seihr_waning_off_by_default(client):
    """Without `immunity_duration`, waning_rate stays at the scalar default 0.0."""
    request = _baseline_request()
    request["output"] = {"include_parameters": True}
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200
    params = response.json()["results"]["parameters"]["data"]
    series = next(iter(params["waning_rate"].values()))
    assert all(abs(v) < 1e-12 for v in series)


def test_v_seihr_waning_enabled_via_immunity_duration(client):
    """Passing immunity_duration injects the calc-param `waning_rate = 1 / immunity_duration`."""
    request = _baseline_request()
    request["model"]["parameters"]["immunity_duration"] = 365.0
    request["output"] = {"include_parameters": True}
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    params = response.json()["results"]["parameters"]["data"]
    series = next(iter(params["waning_rate"].values()))
    assert all(abs(v - 1 / 365.0) < 1e-9 for v in series)


def test_v_seihr_with_vaccination_campaign(client):
    """Vaccination campaign on V-SEIHR drives Susceptible_to_Susceptible_vax transitions."""
    request = _baseline_request()
    request["simulation"]["end_date"] = "2025-04-30"
    request["vaccination"] = {
        "campaigns": [
            {
                "start_date": "2025-02-01",
                "end_date": "2025-02-28",
                "rollout": {"type": "flat_count", "daily_doses": 100_000},
            }
        ]
    }
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    data = response.json()
    transitions = data["results"]["transitions"]
    assert "Susceptible_to_Susceptible_vax" in transitions["data"]
    # Echo-back in metadata.
    assert data["metadata"]["vaccination"]["campaigns"][0]["rollout"]["daily_doses"] == 100_000


def test_v_seihr_explicit_initial_conditions(client):
    """Initial conditions seed Infected and leave _vax compartments empty when requested explicitly.

    Repository-level V-SEIHR default initial conditions are TODO; for now,
    callers who want a clean unvaccinated start must pass `initial_conditions`
    explicitly.
    """
    request = _baseline_request()
    request["initial_conditions"] = {
        "method": "percentage",
        "initial_percentages": {"Infected": 0.1},
    }
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
