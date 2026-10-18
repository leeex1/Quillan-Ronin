# Hugging Face Download Log

Repository: [CrashOverrideX/Quillan-Ronin](https://huggingface.co/CrashOverrideX/Quillan-Ronin)

This is a manually maintained month-to-month record of the Hugging Face download metric. Values marked `approx.` were reported as rounded figures. The current-month value is recorded with the date available.

## Consolidated download ledger

| Period or date | Downloads | Scope | Evidence / precision | Notes |
|---|---:|---|---|---|
| Month before the 78,712 period | ~67,395 | Calendar month | Approximate | User-provided historical total |
| Formerly recorded as ~90,000 | 78,712 | Rolling dashboard snapshot | Screenshot evidence | [May 21 X post](https://x.com/Crashoverride_X/status/2057485072370901328?s=20); this exact figure replaces the earlier ~90k estimate |
| Formerly recorded as ~100,000 | ~144,000 | Calendar month | Source-reported | [June 17 X post](https://x.com/Crashoverride_X/status/2067240810236502479?s=20); this reported May total replaces the earlier ~100k estimate |
| Last month | ~60,000 | Calendar month | Approximate | User-reported month total |
| 2026-05-17 milestone (reported May 21) | ~3,000 | Point-in-time milestone | Source-reported | Post text says the model had “like 3k downloads” four days earlier; not a calendar-month total; excluded from subtotal |
| 2026-06-01 through 2026-06-17 | 29,000 | Partial month | Source-reported | [June 17 X post](https://x.com/Crashoverride_X/status/2067240810236502479?s=20) says “as of today”; not a June final total; excluded from subtotal |
| Current dashboard reading (2026-09-07) | 70,744 | Rolling dashboard snapshot | Verified | Hugging Face rolling “Downloads last month” value |

## Verification status

- **Verified:** Hugging Face’s public model page and model API both report `70,744` for the current rolling last-month metric.
- **Reconciled historical figures:** the former ~90k estimate is now represented by the exact 78,712 snapshot; the former ~100k estimate is now represented by the ~144,000 reported May total; the preceding month is recorded as approximately 67,395.
- **Not independently recoverable from the public API:** the ~67,395 and ~60k monthly figures remain user-reported historical entries.
- The X-post figures in the ledger are preserved as **source-reported evidence**, not treated as direct Hugging Face API measurements. The 78,712 and ~144,000 rows replace, rather than add to, the older ~90k and ~100k estimates.
- Hugging Face documents daily repository history through Publisher Analytics exports for organization repositories on Team/Enterprise plans. That historical export is not exposed by the public model API.
- Because “Downloads last month” is a rolling metric rather than a fixed calendar-month total, future entries should include the exact date checked.

## Running summary

- Listed reconciled subtotal: **~420,851 downloads** across the five primary monthly/snapshot figures.
- Listed reconciled average: **~84,170 downloads/month**.
- Current month versus last month: **+10,744**, approximately **+17.9%**.
- Current month versus the ~144k spike: **-73,256**, approximately **-50.9%**.
- Earlier historical cumulative reference: **~214,000 downloads**. Its date range is unspecified, so it is intentionally excluded from the subtotal above to avoid double-counting.

## Update procedure

At each monthly check, append one row to the table with:

1. The reporting period or date checked.
2. The exact dashboard value when available.
3. `Approximate` when the figure is rounded.
4. A short note identifying whether it is a full month, partial month, spike, or corrected reading.
