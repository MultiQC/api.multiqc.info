# api.multiqc.info

Code for api.multiqc.info, providing run-time information about available updates.

## Running locally

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

I recommend using something like [Postcode](https://marketplace.visualstudio.com/items?itemName=rohinivsenthil.postcode) (VSCode extension) or [httpie](https://httpie.io/) or similar.

When you're done, <kbd>Ctrl</kbd>+<kbd>C</kbd> to exit, then lean up:

```bash
docker compose down
```
