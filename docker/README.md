# Docker Configuration

This directory contains all Docker-related files for tme-quant.

## Files

- `Dockerfile` - Container definition with FFTW pre-installed
- `docker-compose.yml` - Orchestration configuration
- `.dockerignore` - Files excluded from Docker build context

## Usage

From the project root:

```bash
# Build the image
make docker-build

# Run the container
make docker-run

# Stop the container
make docker-stop
```

Or using Docker directly:

```bash
# Build
docker build -f docker/Dockerfile -t tme-quant:latest .

# Run with docker-compose
cd docker
docker-compose up -d
```

## Prerequisites

Before building, you need:
1. **CurveLab** downloaded from https://curvelet.org/download.php
2. Place CurveLab in a location accessible to Docker (e.g., `~/curvelab/` or `/path/to/CurveLab-2.1.x`)
3. Update the volume mount in `docker-compose.yml` to point to your CurveLab location

## Notes

- The Dockerfile builds FFTW automatically (no manual installation needed)
- CurveLab must be mounted at runtime due to licensing
- Image includes pre-built FFTW 2.1.5
- Uses Python 3.11 in a conda environment

