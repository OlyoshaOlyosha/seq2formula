# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com); versions follow semver.

## v1.6.0 - 2026-11-19

### Added
- Bilingual dashboard: English is the canonical UI language with a full Russian translation. The language is detected from the browser on first launch and can be switched in Settings → Interface language; a new language is one locale folder + one entry.
- Power tokens (carried over from the v1.5.x line): expression bases and bare `n^n` under a safety rule (a power is only built from a constant base, a constant exponent, or `n` — no runaway `2^99999`) — the hard tier is now solved 13/13 by the neuro engine.

### Changed
- Inline dismissible error notes instead of alert popups.
- Calmer run badges; no emoji in the UI (icons are inline SVG).

### Fixed
- Horizontal overflow on the Curves tab (graphs no longer escape the viewport).

## v1.5.0 - 2026-11-17

### Added
- Evolution replay: a Play button on the sequence search hero steps through every distinct champion formula the run produced, redrawing the hero chart at each step; click again to stop and restore the final champion.
- Curve playback: the best-fit curve animates as if a pen were drawing it (~2.5 s).
- Server-side saved drawings (My shapes): named drawings persist on the server, available from any browser, with one-click JSON export.
- Snap grid (8/12/16/24/32), eraser and stroke smoothing toggles; drawing limits raised to 12 strokes / 600 points.
- Curve curiosities gallery: finished unsolved runs with error between 0.2% and 10% are collected on the Curiosities tab with a click-to-load.
- New optional floor function in the grammar as a piecewise-linear building block.

### Changed
- Memetic refinement in the neuro engine: after each generation the top formulas get local algebraic refinement (constant nudges, operator swaps, wrapper strips); improvements feed back into training. Hard-tier benchmark: 28/39 → 30/39 runs, 10/13 → 11/13 puzzles — n^2+1 now solves on all seeds, with zero per-generation overhead.

## v1.4.0 - 2026-11-14

### Added
- Curve mode: a second engine fits parametric formulas to 2D point sets. Draw a target shape on canvas (8 presets: line, wave, circle, spiral, zigzag, star, heart...) or build multi-stroke drawings like a smiley, and watch the search match it live with full dashboard statistics.
- Neuro engine (stage 2): reinforcement learning as an alternative to genetic search. A small neural policy (numpy only, no ML framework) assembles formulas token by token and trains live per puzzle via Cross-Entropy Method — 8/9 easy puzzles solved in real time.
- `/neuro` training page: reward/error curves, policy entropy, validity, token probability bars, and the best formula rendered with KaTeX. The main page got a Neuro tab.

### Changed
- Curve search tuning (generic mechanics only): trig-heavy seeds, coherent trig blocks, smoothness fitness penalty, memetic constant refinement — first-ever circle solve (err 2.8%), wave solved, spiral improved from ~4-7% to ~3.2%.
- Engine errors no longer kill the background runner silently; every mode streams its own events over the same WebSocket.
- numpy is now the single agreed extra dependency (curve batch fitness, neuro engine).

## v1.3.0 - 2026-10-27

### Added
- Autopilot (batch mode): run through the entire library automatically, respecting filters and sorting.
- Seed input for reproducible runs: leave empty for a random seed or enter an integer to replay the same evolution.
- Solve notifications: quiet chime and tab-title flash; volume and toggles in Settings.
- JSON export of run history.
- 4 new sequences (Lucas, Fermat, Cullen, Woodall numbers); all sequences capped at 10 terms to prevent overfitting.
- Math verification script that parses reference formulas and automatically expands short sequences.

### Changed
- Dashboard tools: hotkeys (Space, arrows, Esc), clickable table sorting, text filters, status filter (All/Unsolved/Solved/Easy/Hard).
- Unsolved progress is saved on pause/stop and shown immediately in the results table.

### Fixed
- Simplifier rewrote `7 - n` as `-n + 7`, causing a size increase that stalled the shrink phase.

## v1.2.0 - 2026-10-27

### Added
- Custom reference formulas: input your own formula to compare against the search; tabbed layout (Library, My Sequences) and copy-to-clipboard.
- Library statistics summary: overall progress, per-tier completion rates, average generations, top-5 personal records.
- "Keep Searching" button: after a solve, keep running to look for a shorter formula.
- Champion records: attempts and best generation count tracked per puzzle.

### Changed
- Diversity mechanics: fitness sharing against clones, weighted parsimony, anti-stagnation shocks.
- Algebraic simplification (x + x becomes 2x, x * x becomes x^2) for cleaner results.
- Color-coded controls, target sequence plotted side-by-side, speed limit raised to 100 gen/s.
- History saves the exact engine settings of every run, viewable in a tooltip.
- Unsolved runs save their best approximation on pause/stop.

### Fixed
- Speed control overshooting the slider limit due to an OS-level timing issue.
- A correct but longer formula could be ignored when a shorter slightly-wrong one scored better fitness.
- Restarting a puzzle reset the dropdown to "own sequence".
- Tooltip clipping in the dashboard, event-feed text wrapping, port conflicts on server start.

## v1.1.0 - 2026-10-27

### Added
- Grammar v2: square/cube roots, factorials, trigonometry, logarithms, modulo, constants pi and e; the operator picker is split into Basic and Advanced groups.
- 19 new puzzles (library now 41 sequences) with a new T5 tier that requires the advanced operations.
- Settings tab: population size, max generations, mutation rates, speed limits; saved locally between runs.
- Formula plots: a mini-chart of formula vs target on the dashboard, hoverable previews in the history and curiosities tables.
- Rich math rendering of formulas via KaTeX in the event feed and solved notifications.

### Changed
- Roughly 3x faster generations (up to 90-150 gen/s) after rewriting the core evaluation; default speed raised to 30 gen/s.
- Anti-stagnation shocks: fresh random formulas are injected after 60 stagnant generations to keep diversity high.
- Search defaults tuned by automated parameter sweeps over the whole library.
- Clearer per-term accuracy table with plain-language labels.
- "New best" feed messages fire only on real improvements, not on every minor update.

### Fixed
- Chart scaling is no longer stretched by far-out candidates (out-of-window points are hidden).
- Off-by-one in generation counting (a 10,000-generation run now ends on generation 10000).
- The engine could occasionally breed forbidden `n^n` structures.

## v1.0.0 - 2026-10-25

### Added
- Genetic programming engine that finds closed-form formulas behind number sequences using evolution.
- Live dashboard: fitness/error charts, population diversity, per-term accuracy tables.
- Operator picker: choose which operations (addition, subtraction, multiplication, division, power) the engine may use.
- Leaderboard and event feed: live top-5 of the best formulas plus a chronological feed of records, exact hits and formula shrinking.
- Puzzle library: 18 sequences in 3 difficulty tiers.

### Changed
- UI polish: hover tooltips on all metrics, readable chart axis labels, aligned input fields, visually distinct operator selection.

