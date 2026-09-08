# Expert-Routed Harmful Speech Detection App (UI)

Frontend UI for SERML built with Vite, React, and Tailwind CSS.

Node.js 22 or newer is required for the pinned Wrangler deployment tool.

## Setup

```bash
cd "web app/frontend"
npm ci
cp .env.example .env
```

Set backend URL in `.env`:

```bash
VITE_API_BASE_URL=https://api.example.com
```

Production CI injects the direct IAP-protected Cloud Run URL. Users must select
**Connect protected API** once to establish an IAP browser session before the
cross-origin API request can include that session cookie.

`VITE_ENABLE_MOCK_FALLBACK` defaults to `false`. Keep it disabled in production
so an unavailable classifier cannot be mistaken for a real model result.

## Run

```bash
npm run dev
```

## Build

```bash
npm run build
npm run preview
```
