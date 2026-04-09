---
sidebar_position: 2
---

# Production

## Fly.io

The project includes a `fly.toml` configuration. Make sure [flyctl](https://fly.io/docs/hands-on/install-flyctl/) is installed and you're logged in.

```bash
fly auth login
fly deploy
```

## Docker

Build and run the production image:

```bash
docker build -t epydemix-api .
docker run -p 8000:8000 epydemix-api
```

## Environment variables

See [Configuration](/docs/reference/configuration) for all available settings.
