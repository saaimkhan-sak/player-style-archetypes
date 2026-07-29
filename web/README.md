# NHL Player Style Lab

A standalone, Vercel-ready reconstruction of the Streamlit experience.

The app keeps the same five-part exploration model:

- overview
- style glossary
- season explorer
- career explorer
- playoff explorer

It is intentionally isolated in `web/`. The existing `app/` Streamlit project
is not imported, rewritten, or required at runtime.

## Refresh the browser data

From the repository root:

```bash
./.venv/bin/python web/scripts/build_site_data.py
```

## Validate

```bash
cd web
npm run build
```

## Preview

Serve the `web/` directory with any static file server. The app has no runtime
dependencies and lazy-loads only the committed JSON needed for each page.
