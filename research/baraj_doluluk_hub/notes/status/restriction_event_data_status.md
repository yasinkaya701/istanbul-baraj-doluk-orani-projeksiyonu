# Restriction And Outage Event Data Status

Date: 2026-03-11

## Current finding

The official ISKI outage page exists and is backed by an official API endpoint.

Confirmed path:

- public page:
  `https://iski.istanbul/abone-hizmetleri/ariza-kesinti/`
- payload file:
  `/_nuxt/static/.../abone-hizmetleri/ariza-kesinti/payload.js`
- endpoint inside payload:
  `https://iskiapi.iski.gov.tr/api/iski/bolgeselAriza/listesi`

## Technical status

Direct requests to the official endpoint currently return:

- `403 Forbidden`

This means:

- the endpoint exists
- the page is wired to a real official data source
- direct backend access is blocked from simple requests

## Implication for the model

Water cuts / outage events are still a viable feature candidate, but the data
acquisition path is not a simple direct API pull.

Likely acquisition options:

1. browser-mediated capture
2. periodic page scraping if the rendered page exposes table rows
3. official archive or manual export if available later
4. external news or announcement archive as a fallback event timeline

## Professional handling rule

Until a reliable historical feed is available:

- do not fabricate intervention events
- keep the registry template ready
- log every confirmed event source
- treat this as a real data gap, not a hidden assumption
