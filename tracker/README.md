# The 104-Week Plan — tracker

Source for the interactive tracker that accompanies
[`../docs/COMPETENCY_ROADMAP.md`](../docs/COMPETENCY_ROADMAP.md) and
[`../docs/READING_AND_PRACTICE.md`](../docs/READING_AND_PRACTICE.md). The plan data — gate weeks,
success metrics, the reading shelf, the drills, the re-plan triggers — is defined at the top of
`index.html`, so editing the plan means editing those constants.

## Where progress is stored

The page is published as a private Claude Artifact, where it declares the `db` runtime capability
and writes to that artifact's own server-side store under `progress/`:

| Document | Holds |
|---|---|
| `progress/gates` | milestone gates → `{w, at}`, recording the week each was cleared |
| `progress/metrics` | success metrics → `{w, at}` |
| `progress/reading` | book id → `doing` \| `done` |
| `progress/weeks` | week number → `{h, s}` — hours logged and what shipped |

Server-side storage is the point: it survives a cleared browser and follows the reader across
devices, which `localStorage` does not. Two years is too long to trust browser storage with the
only copy.

The page still degrades honestly. It keeps a `localStorage` mirror as a fallback, shows a banner
naming exactly what is and is not being saved when the store cannot be reached, and offers
**Export backup** for the raw JSON.

## Running it outside the artifact

Opening `index.html` directly — from disk or from GitHub Pages — works, but `window.claude` does
not exist there, so every `claude.use()` resolves to nothing and the page runs in local-only mode
with the banner shown. That is the fallback path, not a second supported home: the durable store
exists only in the published artifact.

## Dates

Week 1 begins Monday 28 September 2026; week 104 ends Sunday 24 September 2028. Every phase
boundary lands on a Monday. The current week, the nominal-pace comparison and each gate's
on-time / overdue chip are computed against the reader's real clock, so the page tells them where
they actually are rather than where the plan hoped they would be.
