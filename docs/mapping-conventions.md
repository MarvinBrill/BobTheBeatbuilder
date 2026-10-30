# Mapping conventions from the osu! community

What the osu! forum ([Mapping Discussion](https://osu.ppy.sh/community/forums/56), read
October 2026) and the Ranking Criteria say that a generator can use, and what
BobTheBeatbuilder does with it. The model only learns hit objects; everything below is a
rule-based pass after generation or a checklist for the finished draft.

## Sources

| Source | What it contributes |
| --- | --- |
| [Ranking Criteria, general](https://osu.ppy.sh/wiki/en/Ranking_criteria) | Kiai starts on a sound and never on the first red line; preview point must be set; hitsounds must give feedback (slider ends, reverses, spinner ends too); objects within 2 ms of a 1/16 or 1/12 tick; at least 30 s drain; cut more than 20 % unmapped outro; hit objects, hitsounds and timing must come from direct human input |
| [Ranking Criteria, osu!](https://osu.ppy.sh/wiki/en/Ranking_criteria/osu%21) | Per tier: AR/OD/HP/CS ranges, spinner recovery (Easy 4, Normal 2, Hard 1 beats) and minimum length (4/3/2 beats), overlap rules, distance snap 0.8–1.2× (Easy) / 0.8–1.3× (Normal), streams ≤ 5 notes in Hard, ≤ 3 clicked 1/2 in a row and a 1/1 gap every two bars in a lowest-difficulty Normal, "avoid similar visual spacing for different rhythms" in Hard |
| [Kibbleru – [Guide] Hitsounding!](https://osu.ppy.sh/community/forums/topics/306177) | Claps follow the snare on the 2nd and 4th beat; finishes on cymbal crashes; whistles on downbeats or instead of claps when there is no snare; similar sections share hitsounds |
| [Poll: visually similar spacings in lower difficulties](https://osu.ppy.sh/community/forums/topics/2247525) (164 votes) | 75 % see it as an issue in Easy, 48 % also in Normal: new players follow the closest circle, not the numbers, so a third object close to the first one is read in the wrong order |
| [rhythm/pattern consistency vs. diversity](https://osu.ppy.sh/community/forums/topics/2258211) | Map identical sounds alike, vary every 2–4 repetitions, and use variation to mark real musical change; density follows the song's intensity |
| [Question on RC kiai toggle at end of map](https://osu.ppy.sh/community/forums/topics/2247362) | Ending kiai with an explicit timing point is common practice and expected by modders |
| [Could AI-assisted beatmaps fit into Project Loved?](https://osu.ppy.sh/community/forums/topics/2228051), [using ai for basic hs](https://osu.ppy.sh/community/forums/topics/2254134) | The community rejects AI maps and AI hitsounds for Ranked/Loved; transparency and human final decisions are the minimum. Generated maps stay drafts |

## What generation now does

`backend/app/ml/finishing.py`, after the objects are final (no retraining needed):

- **Kiai** on sustained loud sections (4 s smoothed loudness ≥ −3 dB relative to the loud
  parts, at least 8 s / 16 beats, at most half of the mapped time, and at least 3 dB above
  the rest). It starts and ends on an object head, preferably on a bar line. The first red
  line never toggles kiai; a kiai there is set by a green line at the same time.
- **Preview point** at the first kiai, otherwise at the loudest 10 s.
- **Break periods** wherever osu!lazer's editor would add them (200 ms after the previous
  object, max(450 ms, approach time) before the next, at least 650 ms), so HP no longer
  drains through long rests.
- **Hitsounds** (`hitsounds`: `basic` default, `full`, `off`): claps on beats 2 and 4,
  finishes on kiai starts, phrase starts inside kiai (every 4 bars), the note after a
  spinner and loudness drops (at least 2 beats apart), whistles on downbeats in calm parts
  (`full`: on every downbeat). Slider heads, repeats and tails get their own edge sounds;
  spinners that lead into a drop end with a finish.
- **Volume** per section: calm 50 %, normal 65 %, kiai 80 %; changes shorter than two bars
  are dropped.
- **Spinners** respect the target tier's recovery and minimum length.
- **Easy/Normal targets**: candidate selection adds a readability cost for misleading
  reading order and for spacing outside the tier's distance-snap range.

The preview plays whistle, finish and clap samples from the skin at the timing-point
volume, and spinner end sounds.

## RC check

`backend/app/ranking_check.py` reviews every generated difficulty for its tier (measured
stars, otherwise the target) and stores the findings in the output's quality report. The
studio shows them with editor timestamps and can copy them as modding text. Severity
`problem` marks an RC rule, `warning` a guideline. The check always adds a note that
generated objects, hitsounds and timing cannot be submitted for Ranked or Loved.

## Not done (ideas)

- **Snare/crash detection** from spectral bands instead of beat positions, so claps follow
  the actual drums and syncopated snares.
- **Section-consistent patterns**: detect repeated sections (choruses) and reuse their
  rhythm and hitsounds, varying every 2–4 repetitions; `motif_memory` is a starting point.
- **Spread checks** across several difficulties of one set (drain-time based lowest
  difficulty, no skipped tiers).
