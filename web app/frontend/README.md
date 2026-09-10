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

The production container serves this UI and the API from the same direct
IAP-protected Cloud Run origin. This avoids depending on third-party cookies for
cross-site requests. The Cloudflare deployment redirects users to that protected
origin; local development can still set `VITE_API_BASE_URL` explicitly.

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
