# epydemix WebAPI

REST API for running epidemic simulations on [epydemix](https://github.com/epistorm/epydemix).

## Quick Start

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn app.main:app --reload

# Run tests
uv run pytest
```

## API Endpoints

- `POST /api/v1/simulations` - Run epidemic simulation
- `GET /api/v1/populations` - List available populations
- `GET /api/v1/populations/{name}` - Get population details
- `GET /api/v1/populations/{name}/contacts` - Get contact matrices
- `GET /api/v1/models/presets` - List model presets
- `GET /api/v1/health` - Health check

## Docker

```bash
# Build and run
docker compose up

# Development with hot reload
docker compose --profile dev up api-dev
```

## Examples

```bash
# List available populations
curl http://localhost:8000/api/v1/populations

# Run SIR simulation
curl -X POST http://localhost:8000/api/v1/simulations \
  -H "Content-Type: application/json" \
  -d '{
    "model": {"preset": "SIR"},
    "population": {"name": "United_States"},
    "simulation": {
      "start_date": "2024-01-01",
      "end_date": "2024-03-01",
      "Nsim": 100
    }
  }'
```

## Documentation

API documentation is available at `/api/v1/docs` when running the server.

Most API parameters follow epydemix conventions. Refer to the [epydemix documentation](https://epydemix.readthedocs.io/en/latest/) for details on model parameters, population data, and simulation options.
