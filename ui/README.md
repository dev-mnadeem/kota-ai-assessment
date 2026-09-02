# Web UI

React 19 + Vite front end for the pension tables assistant. It reads `/ready` to
show which snapshot the API is serving, posts questions to `/api/ask`, and shows the
answer with its cited tables and the tool lookups behind it.

```bash
cp .env.example .env   # VITE_API_BASE_URL, default http://127.0.0.1:8000
npm ci
npm run dev            # http://localhost:5173
npm test               # Vitest + Testing Library in jsdom
npm run lint           # oxlint
npm run build          # static bundle in dist/
```

See the repository README for the backend and dataset setup.
