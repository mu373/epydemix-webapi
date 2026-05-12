"""Calculated (expression-valued) parameters in `model.parameters`."""

import pytest

_AGE_GROUP_MAPPING_5 = {
    "0-4": ["0-4"],
    "5-17": ["5-9", "10-14", "15-19"],
    "18-49": ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49"],
    "50-64": ["50-54", "55-59", "60-64"],
    "65+": ["65-69", "70-74", "75+"],
}


def _custom_sir_request(parameters: dict) -> dict:
    """SIR custom model used as a substrate for exercising calculated params.

    Defines `S, I, R` with `S → I` mediated by `transmission_rate` and
    `I → R` spontaneous at `recovery_rate`. Both rate names must resolve
    in `model.parameters` (scalar, list, or calculated).
    """
    return {
        "model": {
            "compartments": ["S", "I", "R"],
            "parameters": parameters,
            "transitions": [
                {"source": "S", "target": "I", "kind": "mediated", "params": ["transmission_rate", "I"]},
                {"source": "I", "target": "R", "kind": "spontaneous", "params": "recovery_rate"},
            ],
        },
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "Nsim": 2,
            "seed": 1,
        },
        "output": {"include_parameters": True},
    }


def test_calculated_param_simple_scalar(client):
    """A calculated scalar resolves to the expected product of source scalars."""
    request = _custom_sir_request(
        {
            "p_h": 0.05,
            "gamma": 0.2,
            "transmission_rate": 0.3,
            "recovery_rate": "(1 - p_h) * gamma",
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"

    params = response.json()["results"]["parameters"]
    series = next(iter(params["data"]["recovery_rate"].values()))
    assert series[0] == pytest.approx(0.95 * 0.2, abs=1e-9)


def test_calculated_param_chained(client):
    """A calculated parameter that depends on another calculated parameter
    is evaluated in topological order, regardless of dict insertion order."""
    request = _custom_sir_request(
        {
            # `recovery_rate` references `intermediate`, which is itself an
            # expression. We list them in reverse order to prove the topo sort
            # doesn't rely on dict order.
            "recovery_rate": "intermediate * 2",
            "intermediate": "gamma * (1 - p_h)",
            "transmission_rate": 0.3,
            "p_h": 0.1,
            "gamma": 0.2,
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    params = response.json()["results"]["parameters"]
    series = next(iter(params["data"]["recovery_rate"].values()))
    assert series[0] == pytest.approx(0.2 * 0.9 * 2, abs=1e-9)


def test_calculated_param_with_age_varying_source(client):
    """A list-valued source flows through the expression as age-varying."""
    request = _custom_sir_request(
        {
            "p_h": [0.05, 0.10, 0.15, 0.20, 0.25],
            "gamma": 0.2,
            "transmission_rate": 0.3,
            "recovery_rate": "(1 - p_h) * gamma",
        }
    )
    request["population"] = {
        "name": "United_States",
        "contacts_source": "prem_2021",
        "age_group_mapping": _AGE_GROUP_MAPPING_5,
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text

    rec = response.json()["results"]["parameters"]["data"]["recovery_rate"]
    expected = [0.95 * 0.2, 0.90 * 0.2, 0.85 * 0.2, 0.80 * 0.2, 0.75 * 0.2]
    age_keys = [k for k in rec if k != "total"]
    # age groups in `rec` follow the resolved population order, which matches
    # _AGE_GROUP_MAPPING_5's insertion order.
    for ag, want in zip(age_keys, expected):
        assert rec[ag][0] == pytest.approx(want, abs=1e-9)


def test_calculated_param_composes_with_balcan(client):
    """A balcan transform on a scalar source produces a time-varying
    calculated parameter via numpy broadcasting."""
    request = _custom_sir_request(
        {
            "p_h": 0.1,
            "gamma": 0.2,
            "transmission_rate": 0.3,
            "recovery_rate": "(1 - p_h) * gamma",
        }
    )
    request["parameter_transforms"] = [
        {
            "target_parameter": "gamma",
            "method": "balcan",
            "max_date": "2024-01-15",
            "min_date": "2024-07-15",
            "max_value": 0.30,
            "min_value": 0.10,
        }
    ]

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text

    rec = response.json()["results"]["parameters"]["data"]["recovery_rate"]
    series = next(iter(rec.values()))
    # The series should not be flat — balcan made `gamma` time-varying, and
    # `(1 - 0.1) * gamma` therefore varies over time too.
    assert max(series) - min(series) > 1e-3


def test_calculated_param_composes_with_age_varying_plus_balcan(client):
    """Source is age-varying AND time-transformed; calculated output is (T, N)."""
    request = _custom_sir_request(
        {
            "p_h": [0.05, 0.10, 0.15, 0.20, 0.25],
            "gamma": 0.2,
            "transmission_rate": 0.3,
            "recovery_rate": "(1 - p_h) * gamma",
        }
    )
    request["population"] = {
        "name": "United_States",
        "contacts_source": "prem_2021",
        "age_group_mapping": _AGE_GROUP_MAPPING_5,
    }
    request["parameter_transforms"] = [
        {
            "target_parameter": "gamma",
            "method": "balcan",
            "max_date": "2024-01-15",
            "min_date": "2024-07-15",
            "max_value": 0.30,
            "min_value": 0.10,
        }
    ]

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text

    rec = response.json()["results"]["parameters"]["data"]["recovery_rate"]
    age_keys = [k for k in rec if k != "total"]
    # Different age groups have different (1-p_h) factors → different series.
    assert rec[age_keys[0]][0] != pytest.approx(rec[age_keys[-1]][0], abs=1e-6)
    # Each series is time-varying.
    for ag in age_keys:
        s = rec[ag]
        assert max(s) - min(s) > 1e-3


def test_calculated_param_undefined_name(client):
    """Referencing a name that is neither defined nor calculated → 422."""
    request = _custom_sir_request(
        {
            "transmission_rate": 0.3,
            "recovery_rate": "(1 - p_h) * gamma",  # neither p_h nor gamma defined
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    assert "undefined name" in response.json()["detail"].lower()


def test_calculated_param_circular_dependency(client):
    """Mutually referencing expressions → 422 naming the cycle."""
    request = _custom_sir_request(
        {
            "transmission_rate": 0.3,
            "a": "b + 1",
            "b": "a + 1",
            "recovery_rate": "a * 0.1",
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "circular" in detail
    assert "a" in detail and "b" in detail


def test_calculated_param_disallowed_call(client):
    """Function calls are rejected by the AST validator."""
    request = _custom_sir_request(
        {
            "x": 0.3,
            "transmission_rate": 0.3,
            "recovery_rate": "abs(x)",
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    assert "Call" in response.json()["detail"]


def test_calculated_param_disallowed_attribute(client):
    """Attribute access is rejected."""
    request = _custom_sir_request(
        {
            "x": 0.3,
            "transmission_rate": 0.3,
            "recovery_rate": "x.real",
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    assert "Attribute" in response.json()["detail"]


def test_calculated_param_disallowed_subscript(client):
    """Subscripts are rejected."""
    request = _custom_sir_request(
        {
            "x": 0.3,
            "transmission_rate": 0.3,
            "recovery_rate": "x[0]",
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    assert "Subscript" in response.json()["detail"]


def test_calculated_param_syntax_error(client):
    """Unparseable expressions surface as 422 with a clear message."""
    request = _custom_sir_request(
        {
            "p_h": 0.1,
            "gamma": 0.2,
            "transmission_rate": 0.3,
            "recovery_rate": "((1 - p_h",  # unbalanced parens
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    assert "recovery_rate" in response.json()["detail"]


def test_calculated_param_transform_target_rejected(client):
    """A `parameter_transforms` entry targeting a calculated name → 422."""
    request = _custom_sir_request(
        {
            "p_h": 0.1,
            "gamma": 0.2,
            "transmission_rate": 0.3,
            "recovery_rate": "(1 - p_h) * gamma",
        }
    )
    request["parameter_transforms"] = [
        {
            "target_parameter": "recovery_rate",
            "method": "scale",
            "start_date": "2024-01-05",
            "end_date": "2024-01-10",
            "factor": 0.5,
        }
    ]
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "recovery_rate" in detail
    assert "calculated parameter" in detail


def test_calculated_param_not_echoed_in_metadata(client):
    """`metadata` carries no new calculated-parameter field; expressions live
    only in the request and surface via `results.parameters` when requested."""
    request = _custom_sir_request(
        {
            "p_h": 0.1,
            "gamma": 0.2,
            "transmission_rate": 0.3,
            "recovery_rate": "(1 - p_h) * gamma",
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    metadata = response.json()["metadata"]
    assert "calculated_parameters" not in metadata
    # The evaluated value is still observable via include_parameters.
    assert "recovery_rate" in response.json()["results"]["parameters"]["data"]


def test_calculated_param_uses_reserved_eigenvalue(client):
    """R0 calibration via the reserved CONTACT_MATRIX_EIGENVALUE_ALL constant.

    Sets `R0` and derives `transmission_rate = R0 * gamma / eigenvalue`. The
    effective transmission_rate in `results.parameters` should match the
    closed-form value computed from the population's overall contact matrix.
    """
    request = _custom_sir_request(
        {
            "R0": 1.5,
            "gamma": 0.1,
            "recovery_rate": "gamma",
            "transmission_rate": "R0 * gamma / CONTACT_MATRIX_EIGENVALUE_ALL",
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text

    # Independently compute the expected eigenvalue from the same population.
    import numpy as np

    from app.services.population_service import _load_population_cached

    pop = _load_population_cached("United_States")
    summed = sum(np.asarray(m, dtype=float) for m in pop.contact_matrices.values())
    eigenvalue = float(np.max(np.abs(np.linalg.eigvals(summed))))
    expected_beta = 1.5 * 0.1 / eigenvalue

    series = next(iter(response.json()["results"]["parameters"]["data"]["transmission_rate"].values()))
    assert series[0] == pytest.approx(expected_beta, rel=1e-6)


def test_calculated_param_reserved_name_collision_rejected(client):
    """A user parameter named like a reserved constant → 422."""
    request = _custom_sir_request(
        {
            "transmission_rate": 0.3,
            "recovery_rate": 0.1,
            "CONTACT_MATRIX_EIGENVALUE_ALL": 5.0,
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "CONTACT_MATRIX_EIGENVALUE_ALL" in detail
    assert "reserved" in detail.lower()


def test_calculated_param_reserved_value_not_in_results(client):
    """Reserved names exist only in the eval namespace, not in `results.parameters`."""
    request = _custom_sir_request(
        {
            "R0": 1.5,
            "gamma": 0.1,
            "recovery_rate": "gamma",
            "transmission_rate": "R0 * gamma / CONTACT_MATRIX_EIGENVALUE_ALL",
        }
    )
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    data = response.json()["results"]["parameters"]["data"]
    assert "CONTACT_MATRIX_EIGENVALUE_ALL" not in data
    # The downstream calculated value, however, *is* stored and visible.
    assert "transmission_rate" in data


def test_seir_vax_end_to_end(client):
    """Small SEIR + vaccinated branch model with one calculated rate.

    Compartments: S, E, I, R, Sv, Ev, Iv, Rv. Vaccinated transmission rate
    is `(1 - VE_S) * transmission_rate`. Force of infection across both
    branches is encoded via parallel mediated transitions on `I` and `Iv`.
    """
    request = {
        "model": {
            "compartments": ["S", "E", "I", "R", "Sv", "Ev", "Iv", "Rv"],
            "parameters": {
                "transmission_rate": 0.3,
                "sigma": 0.2,
                "gamma": 0.1,
                "nu": 0.005,
                "VE_S": 0.5,
                "transmission_rate_v": "(1 - VE_S) * transmission_rate",
            },
            "transitions": [
                {"source": "S",  "target": "Sv", "kind": "spontaneous", "params": ["nu"]},
                {"source": "S",  "target": "E",  "kind": "mediated",    "params": ["transmission_rate", "I"]},
                {"source": "S",  "target": "E",  "kind": "mediated",    "params": ["transmission_rate", "Iv"]},
                {"source": "Sv", "target": "Ev", "kind": "mediated",    "params": ["transmission_rate_v", "I"]},
                {"source": "Sv", "target": "Ev", "kind": "mediated",    "params": ["transmission_rate_v", "Iv"]},
                {"source": "E",  "target": "I",  "kind": "spontaneous", "params": ["sigma"]},
                {"source": "I",  "target": "R",  "kind": "spontaneous", "params": ["gamma"]},
                {"source": "Ev", "target": "Iv", "kind": "spontaneous", "params": ["sigma"]},
                {"source": "Iv", "target": "Rv", "kind": "spontaneous", "params": ["gamma"]},
            ],
        },
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "Nsim": 2,
            "seed": 1,
        },
        "output": {"include_parameters": True},
    }
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"

    params = response.json()["results"]["parameters"]["data"]
    series = next(iter(params["transmission_rate_v"].values()))
    assert series[0] == pytest.approx(0.5 * 0.3, abs=1e-9)
