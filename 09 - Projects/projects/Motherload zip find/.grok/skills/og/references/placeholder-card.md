# The placeholder `og.grok.me` card (plain utility apps)

An app with no `public/og.jpg` unfurls with the hosted placeholder:

```
https://og.grok.me/v1/card.png?host={VITE_PUBLIC_HOSTNAME}&title={APP_NAME}
```

The injector builds that URL — do not paste it into `__root.tsx`.

- Optional theme colour: set `"color": "FF4D2E"` in `src/lib/og/site.json`
  (6-digit hex, `#` optional). The injector appends `&color=` on the placeholder
  URL; a themed URL written into `__root.tsx` is stripped. Custom cards ignore
  `color`.
- On rename, update `APP_NAME` (tab title) **and** `site.json` `title` (share
  card).

Live preview emits the same tags (the preview `X-Forwarded-Host` is a valid
image host). On publish the request `Host` is enough — do **not** write a `.env`
for it. Card pixels and `site.json` reach the unfurl on the **next deploy**
(identity is baked at `vite build`).
