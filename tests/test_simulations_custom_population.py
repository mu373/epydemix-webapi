"""Custom-population path through the simulation endpoint and schema validators."""

CUSTOM_POPULATION_REQUEST = {
    "model": {
        "preset": "SIR",
        "parameters": {"transmission_rate": 0.3, "recovery_rate": 0.1},
    },
    "population": {
        "source": "custom",
        "name": "Toy population",
        "age_groups": {"A": 100, "B": 100},
        "contact_matrices": {"all": [[0.2, 0.3], [0.3, 0.2]]},
    },
    "simulation": {
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "Nsim": 5,
    },
}


def test_run_simulation_with_custom_population(client):
    """End-to-end: a custom 2-group population runs and metadata reflects the request."""
    response = client.post("/api/v1/simulations", json=CUSTOM_POPULATION_REQUEST)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["status"] == "completed"

    pop_meta = data["metadata"]["population"]
    assert pop_meta["source"] == "custom"
    assert pop_meta["name"] == "Toy population"
    assert pop_meta["layers"] == ["all"]
    assert pop_meta["age_groups"] == {"A": 100, "B": 100}
    assert pop_meta["total"] == 200
    assert pop_meta["contacts_source"] is None
    assert pop_meta["age_group_mapping"] is None
    assert pop_meta["contact_matrices"] == {"all": [[0.2, 0.3], [0.3, 0.2]]}

    # Both age-group keys appear in the trajectory data.
    susceptible = data["results"]["compartments"]["data"]["Susceptible"]
    assert "A" in susceptible
    assert "B" in susceptible


def test_custom_population_default_name(client):
    """Omitting `name` falls back to `"Custom Population"`."""
    payload = {
        **CUSTOM_POPULATION_REQUEST,
        "population": {
            "source": "custom",
            "age_groups": {"A": 100, "B": 100},
            "contact_matrices": {"all": [[0.2, 0.3], [0.3, 0.2]]},
        },
    }
    response = client.post("/api/v1/simulations", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["population"]["name"] == "Custom Population"


def test_custom_population_multiple_layers_preserve_order(client):
    """`contact_matrices` key order defines the metadata `layers` order."""
    payload = {
        **CUSTOM_POPULATION_REQUEST,
        "population": {
            "source": "custom",
            "name": "multi-layer",
            "age_groups": {"A": 100, "B": 100},
            "contact_matrices": {
                "home": [[0.10, 0.05], [0.05, 0.10]],
                "work": [[0.15, 0.20], [0.20, 0.15]],
                "school": [[0.05, 0.05], [0.05, 0.05]],
            },
        },
    }
    response = client.post("/api/v1/simulations", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["population"]["layers"] == ["home", "work", "school"]


def test_custom_population_intervention_against_custom_layer(client):
    """Interventions can target a layer the user declared in `contact_matrices`."""
    payload = {
        **CUSTOM_POPULATION_REQUEST,
        "population": {
            "source": "custom",
            "name": "with intervention",
            "age_groups": {"A": 100, "B": 100},
            "contact_matrices": {
                "home": [[0.10, 0.05], [0.05, 0.10]],
                "work": [[0.15, 0.20], [0.20, 0.15]],
            },
        },
        "interventions": [
            {
                "layer_name": "work",
                "start_date": "2024-01-10",
                "end_date": "2024-01-20",
                "reduction_factor": 0.5,
                "name": "lockdown",
            }
        ],
    }
    response = client.post("/api/v1/simulations", json=payload)
    assert response.status_code == 200, response.text
    interventions = response.json()["metadata"]["interventions"]
    assert interventions and interventions[0]["layer_name"] == "work"


def test_custom_population_empty_age_groups_rejected(client):
    """Empty `age_groups` is a 422."""
    payload = {
        **CUSTOM_POPULATION_REQUEST,
        "population": {
            "source": "custom",
            "age_groups": {},
            "contact_matrices": {"all": [[0.2]]},
        },
    }
    response = client.post("/api/v1/simulations", json=payload)
    assert response.status_code == 422
    assert "age_groups" in response.text


def test_custom_population_empty_contact_matrices_rejected(client):
    """Empty `contact_matrices` is a 422."""
    payload = {
        **CUSTOM_POPULATION_REQUEST,
        "population": {
            "source": "custom",
            "age_groups": {"A": 100},
            "contact_matrices": {},
        },
    }
    response = client.post("/api/v1/simulations", json=payload)
    assert response.status_code == 422
    assert "contact_matrices" in response.text


def test_custom_population_dim_mismatch_rejected(client):
    """A matrix whose row count != len(age_groups) is a 422."""
    payload = {
        **CUSTOM_POPULATION_REQUEST,
        "population": {
            "source": "custom",
            "age_groups": {"A": 100, "B": 100, "C": 100},  # 3 groups
            "contact_matrices": {"all": [[0.2, 0.3], [0.3, 0.2]]},  # 2x2
        },
    }
    response = client.post("/api/v1/simulations", json=payload)
    assert response.status_code == 422


def test_custom_population_non_square_matrix_rejected(client):
    """A non-square matrix is a 422."""
    payload = {
        **CUSTOM_POPULATION_REQUEST,
        "population": {
            "source": "custom",
            "age_groups": {"A": 100, "B": 100},
            "contact_matrices": {"all": [[0.2, 0.3, 0.1], [0.3, 0.2, 0.1]]},  # 2x3
        },
    }
    response = client.post("/api/v1/simulations", json=payload)
    assert response.status_code == 422


def test_custom_population_reserved_layer_name_rejected(client):
    """Layer name `"overall"` is reserved by epydemix and rejected at request time."""
    payload = {
        **CUSTOM_POPULATION_REQUEST,
        "population": {
            "source": "custom",
            "age_groups": {"A": 100, "B": 100},
            "contact_matrices": {"overall": [[0.2, 0.3], [0.3, 0.2]]},
        },
    }
    response = client.post("/api/v1/simulations", json=payload)
    assert response.status_code == 422
    assert "overall" in response.text


def test_legacy_payload_without_source_field_still_works(client):
    """Existing `{"population": {"name": "United_States"}}` payloads keep working."""
    payload = {
        "model": {
            "preset": "SIR",
            "parameters": {"transmission_rate": 0.3, "recovery_rate": 0.1},
        },
        "population": {"name": "United_States"},  # no `source` field
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "Nsim": 3,
        },
    }
    response = client.post("/api/v1/simulations", json=payload)
    assert response.status_code == 200, response.text
    pop_meta = response.json()["metadata"]["population"]
    assert pop_meta["source"] == "builtin"
    assert pop_meta["name"] == "United_States"
    # Builtin path resolves contacts_source.
    assert pop_meta["contacts_source"] is not None
    # Builtin path also fills contact_matrices from the loaded population.
    assert pop_meta["contact_matrices"], "expected at least one contact-matrix layer"


def test_simulation_request_routes_to_custom_branch():
    """Direct schema instantiation: discriminator routes payload to CustomPopulationConfig."""
    from app.api.v1.schemas.simulation import CustomPopulationConfig, SimulationRequest

    req = SimulationRequest.model_validate(CUSTOM_POPULATION_REQUEST)
    assert isinstance(req.population, CustomPopulationConfig)
    assert req.population.source == "custom"
    assert list(req.population.age_groups.keys()) == ["A", "B"]


def test_homogeneous_sir_matches_ode_solution(client):
    """Numeric correctness: a single-group custom population with a 1x1 contact
    matrix must reproduce the homogeneous SIR ODE in expectation. The stochastic
    median (Nsim=100, seed=42, dt=0.25) is compared against an RK4 integration
    of the deterministic equations from the same initial conditions.

    Tolerances are sized at ~2x the empirical errors measured at this config so
    seed-noise / environment drift have headroom without dulling regression
    detection.
    """
    import numpy as np

    BETA, GAMMA, N = 0.3, 0.1, 100_000

    payload = {
        "model": {
            "preset": "SIR",
            "parameters": {"transmission_rate": BETA, "recovery_rate": GAMMA},
        },
        "population": {
            "source": "custom",
            "name": "Homogeneous N=100k",
            "age_groups": {"A": N},
            "contact_matrices": {"all": [[1.0]]},
        },
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-06-01",
            "Nsim": 100,
            "dt": 0.25,
            "seed": 42,
        },
    }

    response = client.post("/api/v1/simulations", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "completed"

    comp = data["results"]["compartments"]["data"]
    S_med = np.array(comp["Susceptible"]["A"]["0.5"], dtype=float)
    I_med = np.array(comp["Infected"]["A"]["0.5"], dtype=float)
    T = len(S_med)

    # RK4 of dS=-bSI/N, dI=bSI/N-gI, dR=gI from the same IC the simulator used.
    S0, I0, R0_count = float(S_med[0]), float(I_med[0]), float(N - S_med[0] - I_med[0])

    def deriv(s, i, _r):
        infection = BETA * s * i / N
        return -infection, infection - GAMMA * i, GAMMA * i

    h = 0.05  # RK4 step in days; ODE error here is negligible
    s, i, r = S0, I0, R0_count
    S_ode = np.empty(T)
    I_ode = np.empty(T)
    S_ode[0], I_ode[0] = s, i
    for day in range(1, T):
        steps = int(round(1.0 / h))
        for _ in range(steps):
            k1 = deriv(s, i, r)
            k2 = deriv(s + h * k1[0] / 2, i + h * k1[1] / 2, r + h * k1[2] / 2)
            k3 = deriv(s + h * k2[0] / 2, i + h * k2[1] / 2, r + h * k2[2] / 2)
            k4 = deriv(s + h * k3[0], i + h * k3[1], r + h * k3[2])
            s += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            i += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            r += h * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        S_ode[day], I_ode[day] = s, i

    peak_day_sim = int(np.argmax(I_med))
    peak_day_ode = int(np.argmax(I_ode))
    peak_height_sim = float(I_med[peak_day_sim])
    peak_height_ode = float(I_ode[peak_day_ode])
    final_size_sim = float(N - S_med[-1])
    final_size_ode = float(N - S_ode[-1])

    # Peak timing within ±2 days (measured shift at this config: +1 day).
    assert abs(peak_day_sim - peak_day_ode) <= 2, (
        f"peak day mismatch: sim={peak_day_sim}, ode={peak_day_ode}"
    )
    # Peak height within 3% (measured: ~1.3%).
    peak_err = abs(peak_height_sim - peak_height_ode) / peak_height_ode
    assert peak_err < 0.03, (
        f"peak height mismatch: sim={peak_height_sim:.0f}, ode={peak_height_ode:.0f}, err={peak_err:.2%}"
    )
    # Final epidemic size within 1% (measured: ~0.27%). Most stable invariant.
    final_err = abs(final_size_sim - final_size_ode) / final_size_ode
    assert final_err < 0.01, (
        f"final size mismatch: sim={final_size_sim:.0f}, ode={final_size_ode:.0f}, err={final_err:.2%}"
    )
