"""Utilities for parsing and manipulating column names."""


def parse_column_name(col: str, known_bases: list[str]) -> tuple[str, str]:
    """Parse a column name into (base_name, suffix).

    Handles column names like 'Infected_0-4' or 'Susceptible_to_Infected_total'
    by matching against known base names and extracting the suffix.

    Parameters
    ----------
    col : str
        Column name to parse (e.g., 'Infected_0-4', 'S_to_I_total').
    known_bases : list of str
        List of known base names to match against. Longer names are tried
        first to handle cases like 'Susceptible_to_Infected' vs 'Susceptible'.

    Returns
    -------
    tuple of (str, str)
        A tuple of (base_name, suffix). For 'Infected_0-4' returns
        ('Infected', '0-4'). If no known base matches, splits on last
        underscore. If no underscore exists, returns (col, 'total').

    Examples
    --------
    >>> parse_column_name('Infected_0-4', ['Susceptible', 'Infected', 'Recovered'])
    ('Infected', '0-4')
    >>> parse_column_name('S_to_I_total', ['S_to_I', 'I_to_R'])
    ('S_to_I', 'total')
    >>> parse_column_name('Unknown_value', [])
    ('Unknown', 'value')
    """
    # Try to match against known base names (longest first to handle transitions)
    for base in sorted(known_bases, key=len, reverse=True):
        if col.startswith(base + "_"):
            suffix = col[len(base) + 1 :]
            return base, suffix
    # Fallback: split on last underscore
    if "_" in col:
        idx = col.rfind("_")
        return col[:idx], col[idx + 1 :]
    return col, "total"
