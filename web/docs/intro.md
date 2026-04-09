---
slug: /
sidebar_position: 1
---

# Introduction

epydemix Web API is a REST API for running epidemic simulations powered by [epydemix](https://github.com/epistorm/epydemix).

## Overview

The API exposes epydemix's simulation engine over HTTP. You can run compartmental models (SIR, SEIR, SIS) against built-in population data by sending a single JSON request. No Python environment required on the client side.

## How it works

1. Pick a model preset and population
2. POST a simulation request with your parameters
3. Get back compartment trajectories for each simulation run

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | [`/api/v1/simulations`](/api-reference#tag/simulations/POST/api/v1/simulations) | Run a simulation |
| GET | [`/api/v1/populations`](/api-reference#tag/populations/GET/api/v1/populations) | List available populations |
| GET | [`/api/v1/populations/{name}`](/api-reference#tag/populations/GET/api/v1/populations/{name}) | Get population details |
| GET | [`/api/v1/populations/{name}/contacts`](/api-reference#tag/populations/GET/api/v1/populations/{name}/contacts) | Get contact matrices |
| GET | [`/api/v1/models/presets`](/api-reference#tag/model-presets/GET/api/v1/models/presets) | List model presets |
| GET | [`/api/v1/health`](/api-reference#tag/health/GET/api/v1/health) | Health check |

The [API Reference](/api-reference) has interactive docs where you can explore schemas and send requests directly.
