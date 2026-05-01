"""Age-varying base parameters and `parameter_transforms` (balcan / scale / override)."""

import pytest

_AGE_GROUP_MAPPING_5 = {
    "0-4": ["0-4"],
    "5-17": ["5-9", "10-14", "15-19"],
    "18-49": ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49"],
    "50-64": ["50-54", "55-59", "60-64"],
    "65+": ["65-69", "70-74", "75+"],
}


def test_simulation_with_parameter_override(client):
    """Test simulation with parameter override (now via parameter_transforms)."""
    request = {
        "model": {
            "preset": "SIR",
        },
        "population": {
            "name": "United_States",
        },
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "Nsim": 5,
        },
        "parameter_transforms": [
            {
                "target_parameter": "transmission_rate",
                "method": "override",
                "start_date": "2024-01-15",
                "end_date": "2024-01-30",
                "value": 0.15,
            }
        ],
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_simulation_age_varying_parameters_preset(client):
    """Age-varying parameters via list values on a preset model."""
    request = {
        "model": {
            "preset": "SIR",
            "parameters": {
                "transmission_rate": [0.35, 0.35, 0.35, 0.30, 0.25],
                "recovery_rate": [0.10, 0.10, 0.10, 0.08, 0.06],
            },
        },
        "population": {
            "name": "United_States",
            "contacts_source": "prem_2021",
            "age_group_mapping": _AGE_GROUP_MAPPING_5,
        },
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "Nsim": 3,
        },
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"


def test_simulation_age_varying_parameters_custom(client):
    """Age-varying parameters via list values on a custom model."""
    request = {
        "model": {
            "compartments": ["S", "I", "R"],
            "parameters": {
                "beta": [0.35, 0.35, 0.30, 0.25, 0.20],
                "gamma": 0.1,
            },
            "transitions": [
                {"source": "S", "target": "I", "kind": "mediated", "params": ["beta", "I"]},
                {"source": "I", "target": "R", "kind": "spontaneous", "params": "gamma"},
            ],
        },
        "population": {
            "name": "United_States",
            "contacts_source": "prem_2021",
            "age_group_mapping": _AGE_GROUP_MAPPING_5,
        },
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "Nsim": 3,
        },
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"


def test_simulation_age_varying_length_mismatch(client):
    """Length of an age-varying list must match population num_groups."""
    request = {
        "model": {
            "preset": "SIR",
            "parameters": {"transmission_rate": [0.3, 0.2]},  # only 2 entries
        },
        "population": {
            "name": "United_States",
            "contacts_source": "prem_2021",
            "age_group_mapping": _AGE_GROUP_MAPPING_5,  # 5 groups
        },
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "Nsim": 2,
        },
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "transmission_rate" in detail
    assert "5" in detail


def test_simulation_balcan_transform(client):
    """Balcan seasonality runs end-to-end."""
    request = {
        "model": {"preset": "SIR"},
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "Nsim": 3,
        },
        "parameter_transforms": [
            {
                "target_parameter": "transmission_rate",
                "method": "balcan",
                "max_date": "2024-01-15",
                "min_date": "2024-07-15",
                "max_value": 0.35,
                "min_value": 0.15,
            }
        ],
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"


def test_simulation_scale_transform(client):
    """Scaling window runs end-to-end."""
    request = {
        "model": {"preset": "SIR"},
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "Nsim": 3,
        },
        "parameter_transforms": [
            {
                "target_parameter": "transmission_rate",
                "method": "scale",
                "start_date": "2024-02-01",
                "end_date": "2024-02-15",
                "factor": 0.5,
            }
        ],
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"


def test_simulation_age_varying_plus_balcan(client):
    """Age-varying base parameter composes with balcan seasonality.

    Exercises the (N,) -> (T, N) path in apply_transform_to_parameter that
    upstream's seasonality builder does not handle.
    """
    request = {
        "model": {
            "preset": "SIR",
            "parameters": {
                "transmission_rate": [0.35, 0.35, 0.30, 0.25, 0.20],
            },
        },
        "population": {
            "name": "United_States",
            "contacts_source": "prem_2021",
            "age_group_mapping": _AGE_GROUP_MAPPING_5,
        },
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "Nsim": 3,
        },
        "parameter_transforms": [
            {
                "target_parameter": "transmission_rate",
                "method": "balcan",
                "max_date": "2024-01-15",
                "min_date": "2024-07-15",
                "max_value": 0.35,
                "min_value": 0.15,
            }
        ],
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"


def test_simulation_override_age_varying_value(client):
    """Override with a per-age-group list value runs end-to-end."""
    request = {
        "model": {"preset": "SIR"},
        "population": {
            "name": "United_States",
            "contacts_source": "prem_2021",
            "age_group_mapping": _AGE_GROUP_MAPPING_5,
        },
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "Nsim": 3,
        },
        "parameter_transforms": [
            {
                "target_parameter": "transmission_rate",
                "method": "override",
                "start_date": "2024-02-01",
                "end_date": "2024-02-15",
                "value": [0.10, 0.12, 0.10, 0.08, 0.06],
            }
        ],
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"


def _combined_request(transforms_in_order):
    return {
        "model": {"preset": "SIR"},
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "Nsim": 3,
            "seed": 42,
        },
        "parameter_transforms": transforms_in_order,
    }


def test_simulation_transforms_combined(client):
    """Balcan + scale + override on the same parameter runs end-to-end."""
    request = _combined_request(
        [
            {
                "target_parameter": "transmission_rate",
                "method": "balcan",
                "max_date": "2024-01-15",
                "min_date": "2024-07-15",
                "max_value": 0.35,
                "min_value": 0.15,
            },
            {
                "target_parameter": "transmission_rate",
                "method": "scale",
                "start_date": "2024-02-01",
                "end_date": "2024-02-15",
                "factor": 0.5,
            },
            {
                "target_parameter": "transmission_rate",
                "method": "override",
                "start_date": "2024-02-05",
                "end_date": "2024-02-10",
                "value": 0.05,
            },
        ]
    )

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"


def test_simulation_transforms_order_independence_for_overrides(client):
    """Overrides win for their window regardless of position in the transforms list."""
    multiplicative = [
        {
            "target_parameter": "transmission_rate",
            "method": "balcan",
            "max_date": "2024-01-15",
            "min_date": "2024-07-15",
            "max_value": 0.35,
            "min_value": 0.15,
        },
        {
            "target_parameter": "transmission_rate",
            "method": "scale",
            "start_date": "2024-02-01",
            "end_date": "2024-02-15",
            "factor": 0.5,
        },
    ]
    override = {
        "target_parameter": "transmission_rate",
        "method": "override",
        "start_date": "2024-02-05",
        "end_date": "2024-02-10",
        "value": 0.05,
    }

    response_a = client.post(
        "/api/v1/simulations", json=_combined_request(multiplicative + [override])
    )
    response_b = client.post(
        "/api/v1/simulations", json=_combined_request([override] + multiplicative)
    )

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert (
        response_a.json()["results"]["compartments"] == response_b.json()["results"]["compartments"]
    )


def test_simulation_transform_undefined_target_parameter(client):
    """An unknown target_parameter surfaces as a 422 with a clear message."""
    request = {
        "model": {"preset": "SIR"},
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "Nsim": 2,
        },
        "parameter_transforms": [
            {
                "target_parameter": "no_such_param",
                "method": "scale",
                "start_date": "2024-01-05",
                "end_date": "2024-01-10",
                "factor": 0.5,
            }
        ],
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422
    assert "no_such_param" in response.json()["detail"]


def test_simulation_transform_validation_error_missing_field(client):
    """A method-required field is checked by the discriminated union (422)."""
    request = {
        "model": {"preset": "SIR"},
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "Nsim": 2,
        },
        "parameter_transforms": [
            # method=balcan but missing max_date/max_value/min_value
            {"target_parameter": "transmission_rate", "method": "balcan"}
        ],
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422


def test_simulation_transform_invalid_window_scale(client):
    """`scale` rejects end_date < start_date with 422."""
    request = {
        "model": {"preset": "SIR"},
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "Nsim": 2,
        },
        "parameter_transforms": [
            {
                "target_parameter": "transmission_rate",
                "method": "scale",
                "start_date": "2024-02-15",
                "end_date": "2024-02-01",
                "factor": 0.5,
            }
        ],
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422


def test_simulation_transform_invalid_window_override(client):
    """`override` rejects end_date < start_date with 422."""
    request = {
        "model": {"preset": "SIR"},
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "Nsim": 2,
        },
        "parameter_transforms": [
            {
                "target_parameter": "transmission_rate",
                "method": "override",
                "start_date": "2024-02-15",
                "end_date": "2024-02-01",
                "value": 0.1,
            }
        ],
    }

    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 422


def test_simulation_include_parameters_off_by_default(client):
    """`output.include_parameters` defaults to false; `results.parameters` is null."""
    request = {
        "model": {"preset": "SIR"},
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "Nsim": 2,
            "seed": 1,
        },
    }
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200
    assert response.json()["results"].get("parameters") is None


def test_simulation_include_parameters_emits_baked_in_overrides(client):
    """With include_parameters=true, the response shows transmission_rate dropping
    to the override value inside the override window and returning to baseline outside."""
    request = {
        "model": {
            "preset": "SIR",
            "parameters": {"transmission_rate": 0.3, "recovery_rate": 0.1},
        },
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "Nsim": 2,
            "seed": 1,
        },
        "parameter_transforms": [
            {
                "target_parameter": "transmission_rate",
                "method": "override",
                "start_date": "2024-01-10",
                "end_date": "2024-01-15",
                "value": 0.05,
            }
        ],
        "output": {"include_parameters": True},
    }
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200

    params = response.json()["results"]["parameters"]
    assert params is not None
    assert "transmission_rate" in params["data"]
    # Pick any age group; arrays are broadcast so all groups carry the same scalar.
    age_group = next(iter(params["data"]["transmission_rate"]))
    series = params["data"]["transmission_rate"][age_group]
    dates = params["dates"]

    # Outside the override window, value is the baseline 0.3.
    idx_jan_5 = dates.index("2024-01-05")
    assert series[idx_jan_5] == pytest.approx(0.3, abs=1e-9)

    # Inside [Jan 10, Jan 15], the override 0.05 is baked in.
    idx_jan_12 = dates.index("2024-01-12")
    assert series[idx_jan_12] == pytest.approx(0.05, abs=1e-9)

    # After the window, back to baseline.
    idx_jan_20 = dates.index("2024-01-20")
    assert series[idx_jan_20] == pytest.approx(0.3, abs=1e-9)


def test_simulation_metadata_echoes_transforms_and_interventions(client):
    """`SimulationMetadata` should echo back `parameter_transforms` and `interventions`."""
    request = {
        "model": {"preset": "SIR"},
        "population": {"name": "United_States"},
        "simulation": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "Nsim": 2,
            "seed": 1,
        },
        "interventions": [
            {
                "layer_name": "school",
                "start_date": "2024-01-05",
                "end_date": "2024-01-10",
                "reduction_factor": 0.3,
            }
        ],
        "parameter_transforms": [
            {
                "target_parameter": "transmission_rate",
                "method": "scale",
                "start_date": "2024-01-05",
                "end_date": "2024-01-10",
                "factor": 0.5,
            }
        ],
    }
    response = client.post("/api/v1/simulations", json=request)
    assert response.status_code == 200

    metadata = response.json()["metadata"]
    assert metadata["interventions"] is not None
    assert metadata["interventions"][0]["layer_name"] == "school"
    assert metadata["parameter_transforms"] is not None
    assert metadata["parameter_transforms"][0]["method"] == "scale"


def test_simulation_transform_no_aliasing():
    """Two `scale` transforms on the same parameter must compose multiplicatively (not just last-write).

    Catches an aliasing regression where a later transform overwrites the prior
    step's stored array instead of multiplying into it.
    """
    from app.api.v1.schemas.simulation import (
        ModelConfig,
        PopulationConfig,
        ScaleTransform,
        SimulationConfig,
    )
    from app.services.simulation_service import (
        apply_parameter_transforms,
        create_model,
        setup_population,
    )

    model, _ = create_model(ModelConfig(preset="SIR", parameters={"transmission_rate": 1.0}))
    setup_population(model, PopulationConfig(name="United_States"))

    sim_cfg = SimulationConfig(start_date="2024-01-01", end_date="2024-01-10", Nsim=1)
    transforms = [
        ScaleTransform(
            target_parameter="transmission_rate",
            method="scale",
            start_date="2024-01-01",
            end_date="2024-01-10",
            factor=2.0,
        ),
        ScaleTransform(
            target_parameter="transmission_rate",
            method="scale",
            start_date="2024-01-01",
            end_date="2024-01-10",
            factor=3.0,
        ),
    ]
    apply_parameter_transforms(model, list(transforms), sim_cfg)

    # Final stored value should be baseline * 2 * 3 = 6 across the window.
    final = model.get_parameter("transmission_rate")
    assert hasattr(final, "__len__")
    assert all(abs(v - 6.0) < 1e-9 for v in final)
