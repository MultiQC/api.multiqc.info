# api.multiqc.info

Code for [api.multiqc.info](https://api.multiqc.info), providing run-time information about available updates.

## Introduction

The API is a simple tool to provide a metadata endpoint for MultiQC runs.
Currently, there are the following endpoints that are used:

### `/version`

- Information about the latest available release
    - MultiQC uses this to print a log message advising if the current version is out of date, with information about
      how to upgrade.
- _[Planned]_: Broadcast messages
    - Can be used to announce arbitrary information, such as critical changes.
    - No usage currently anticipated, this is mostly a future-proofing tool.
- _[Planned]_: Module-specific warnings
    - Warnings scoped to module and MultiQC version
    - Will allow MultiQC to notify end users via the log if the module that they are running has serious bugs or errors.

### `/downloads`

- MultiQC package downloads across multiple sources, and, when available, different versions:
    - [PyPI](https://pypi.org/project/multiqc) (additionally, split by version)
    - [BioConda](https://bioconda.github.io/recipes/multiqc) (additionally, split by version)
    - [DockerHub](https://hub.docker.com/r/ewels/multiqc)
    - [GitHub clones](https://github.com/ewels/MultiQC/graphs/traffic)
    - [BioContainers (AWS mirror)](https://api.us-east-1.gallery.ecr.aws/getRepositoryCatalogData)

## Logged metrics

MultiQC supplies _some_ information to the API when it requests this endpoint.
This is used to gather metrics on MultiQC usage and tailor development efforts.

Currently, it reports:

- MultiQC version
- _[Planned]_: Python version
- _[Planned]_: Operating system (linux|osx|windows|unknown)
- _[Planned]_: Installation method (pip|conda|docker|unknown)
- _[Planned]_: CI environment (GitHub Actions|none)

No identifying information is collected. No IPs are logged, no information about what MultiQC is being used for or
where, no sample data or metadata is transferred. All code in both MultiQC and this API is open source and can be
inspected.

This version check can be disabled by adding `no_version_check: true` to your MultiQC config (
see [docs](https://multiqc.info/docs/getting_started/config/#checks-for-new-versions)).

The request uses a very short timeout (2 seconds) and fails silently if MultiQC has no internet connection or an
unexpected response is returned.

## Production deployment

A docker image is available for the app here:

```
ghcr.io/multiqc/apimultiqcinfo:latest
```

## Development

### Local build

> **Note:**
> These instructions are intended for local development work, not a production deployment.

Create an `.env` file and replace the `xxx`s with random strings.
Set `UVICORN_RELOAD=--reload` to enable hot-reloading when you save files.

```bash
cp .env.example .env
```

Then, use docker compose to launch the app:

```bash
docker compose up
```

The API should now be available at <http://localhost:8008/>

I recommend using something
like [Postcode](https://marketplace.visualstudio.com/items?itemName=rohinivsenthil.postcode) (VSCode extension)
or [httpie](https://httpie.io/) or similar.

When you're done, <kbd>Ctrl</kbd>+<kbd>C</kbd> to exit, then lean up:

```bash
docker compose down
```

### Dependencies

To add a dependency, add it to the `pyproject.toml` file and then compile the requirements:

```sh
uv pip compile pyproject.toml -o requirements.txt
uv pip compile pyproject.toml --extra dev -o requirements-dev.txt
```


