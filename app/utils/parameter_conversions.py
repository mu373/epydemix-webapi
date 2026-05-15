"""Preset-scoped parameter conversions: period→rate and R0→β.

Lets users supply friendlier source inputs (``infectious_period`` instead of
``recovery_rate``, ``R0`` instead of ``transmission_rate``) on opted-in
presets. Each conversion is a ``(derived, source, expression)`` triple; when
the source is present in the model and the derived is not, the resolver
injects the expression as a calculated parameter.

Scoped to presets that declare an opt-in list. Custom models retain the
existing explicit-parameter contract: name-based collisions would break the
custom-model surface, so nothing implicit is injected when ``preset`` is
``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterConversion:
    """One preset-canonical way to express a model parameter via a friendlier source.

    Attributes
    ----------
    source : str
        The user-facing scalar (e.g. ``infectious_period``, ``R0``) that the
        user supplies instead of the derived rate-form value.
    expression : str
        The calc-param expression injected as ``<derived> = <expression>``
        when ``source`` is present in ``model.parameters``. Must reference
        ``source`` (and possibly other names available at calc-eval time).

    The derived name is the dict key in ``PARAMETER_CONVERSIONS``; it is not
    duplicated here.
    """

    source: str
    expression: str


# Single source of truth for preset-canonical conversions. Keyed by the DERIVED
# parameter name (i.e. what gets injected into model.parameters as a calc-param).
# Adding a new conversion = one dict entry; presets opt in by listing the
# derived name in their registry entry.
#
# CONTACT_MATRIX_EIGENVALUE_ALL is a reserved name injected by the calc-param
# evaluator from the contact matrix; not stored on the model.
PARAMETER_CONVERSIONS: dict[str, ParameterConversion] = {
    "incubation_rate": ParameterConversion("incubation_period", "1 / incubation_period"),
    "recovery_rate": ParameterConversion("infectious_period", "1 / infectious_period"),
    "hosp_recovery_rate": ParameterConversion(
        "hospitalization_duration", "1 / hospitalization_duration"
    ),
    "waning_rate": ParameterConversion("immunity_duration", "1 / immunity_duration"),
    "transmission_rate": ParameterConversion(
        "R0", "R0 * recovery_rate / CONTACT_MATRIX_EIGENVALUE_ALL"
    ),
}


def resolve_parameter_conversions(
    model_parameters: dict[str, Any],
    user_scalar_names: set[str],
    enabled_conversions: list[str],
) -> dict[str, str]:
    """Inject calc-params for source scalars the user supplied on opted-in presets.

    Called after preset construction (so preset defaults are already in
    ``model_parameters``) and before calc-param evaluation. Mutates
    ``model_parameters`` in place (drops the source scalar when the user
    overrides the derived directly) and returns a dict of derived-name →
    expression to merge into the pipeline's ``expr_params``.

    Parameters
    ----------
    model_parameters : dict
        ``model.parameters`` mutated in place. Source scalars are popped when
        the user supplies the derived value directly.
    user_scalar_names : set[str]
        Names of scalar/list parameters the user supplied in the request.
        Distinguishes preset defaults from user inputs; preset defaults alone
        do not satisfy the "user passed the derived" precedence rule.
    enabled_conversions : list[str]
        DERIVED parameter names the preset opts into. Each must be a key in
        ``PARAMETER_CONVERSIONS``. Custom models pass ``[]`` to opt out.

    Returns
    -------
    dict[str, str]
        Derived name → expression. Empty when nothing fires.

    Precedence per derived entry ``conv = PARAMETER_CONVERSIONS[name]``:

    1. If ``name`` is in ``user_scalar_names`` (the user passed the derived
       parameter directly), drop ``conv.source`` from ``model_parameters`` and
       emit no calc-param. The user's scalar stands.
    2. Else if ``conv.source`` is present in ``model_parameters``, emit
       ``name = conv.expression``.
    3. Else do nothing.

    Disabling a flow (e.g. no waning) is handled by the preset: ship a scalar
    default (``waning_rate: 0.0``) and don't define the source. Users enable
    by passing the source (``immunity_duration: 365``), which triggers the
    calc-param injection and overrides the scalar default. No ``None``-as-
    disable handling needed here.
    """
    new_calc_params: dict[str, str] = {}
    for derived_name in enabled_conversions:
        conv = PARAMETER_CONVERSIONS[derived_name]
        if derived_name in user_scalar_names:
            model_parameters.pop(conv.source, None)
            continue
        if conv.source not in model_parameters:
            continue
        new_calc_params[derived_name] = conv.expression
    return new_calc_params
