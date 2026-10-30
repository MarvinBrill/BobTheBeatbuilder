# Generation quality: song-long streams, pauses and spinners

## Case: "Jiyagi – Self Aware", target 6.2 ★, model `5668392b…` (6 epochs, best epoch 3)

| | generated | 6 ★ training maps (median) |
|---|---|---|
| tempo used | 99.4 BPM (audio fallback) | 186 BPM |
| objects / s | 6.4, constant from 12 s to 136 s | 4.5, follows the music |
| gaps ≤ 1/4 beat | 96 % – one run of 860 objects | 23 % |
| stream share | 96 % | 5 % |
| spacing (median) | 5 px | 134 px (jumps 171 px) |
| sliders | 4 % | 36 % |
| gaps ≥ 1 beat, spinners | 0, 0 | – |
| stars | 3.26 | 6.0 |

The audio has clear dynamics (intro and 76–96 s about 6–8 dB below the drops at 40 s,
96 s and 120 s); the map ignored them completely.

## Causes

1. **Half tempo.** The learned beat detection was not confident, so librosa's beat tracker
   decided the tempo. Its 120 BPM prior reported 99.4 BPM for a ~199 BPM song (onset
   autocorrelation is equally strong at 302 ms and 604 ms). Every rhythm class and the tempo
   condition are measured in beats, so 1/2 jumps at 199 BPM became "1/4 streams" for the
   model, and it moved them like streams (stacked, 5 px).
2. **Calibration amplified it.** Stars were low because *aim* was missing (aim 1.34 vs
   speed 1.79). The only levers were the stars condition (raised to 11.9, far outside the
   trained range) and `tempo_bias` (raised to +2, its maximum). `tempo_bias` adds
   `-bias·log2(gap)` to every gap choice in the whole song: +4 for every 1/4 gap.
3. **Pattern sampler without a song-level view.** `choose_next_time` combined the gap head,
   onset evidence and `tempo_bias`. The model's density head (`prediction[:, 2]`, trained on
   each reference map's 4 s object rate including rests) was predicted but never used in
   pattern mode, and loudness played no role. Teacher forcing makes "repeat the previous gap"
   the gap head's strongest cue; sampling at `rhythm_temperature` 0.6 sharpens it further, so
   one 1/4 run can continue for the rest of the song.
4. **No spinners.** Spinners are dropped from the training sequences
   (`Beatmap.ignored`) and the generator had no way to place one.
5. **Validation did not show it.** Teacher-forced rhythm accuracy (86.7 %) rewards copying
   the previous gap. Free validation already generated 2–3× the reference object count on
   one song, but no metric reported over-mapping or run length. Free validation also uses the
   reference timing, so tempo detection and calibration were never tested.

More epochs alone would not fix this: validation loss was best at epoch 3 and the rhythm
group at epoch 1; causes 1–4 are in the generation pipeline.

## Changes (no retraining needed)

- `ml/tempo_octave.py`, `ml/audio.py`: the audio fallback compares the onset envelope's
  periodicity at the reported tempo and at its double/half. Only when the other octave is at
  least 80 % as periodic does a tempo prior decide; the prior is the median tempo of training
  maps at the target stars (186 BPM at 6 ★, `star_profiles`), default 175 BPM.
- `ml/rhythm_guidance.py`, `choose_next_time`: soft costs per candidate gap from
  - the density head, scaled by **dynamics** (every 6 dB below the loud sections halves the
    target density; floor 25 %);
  - a gap more than 2× denser than the local mean gap (bursts stay free);
  - the realised rate of the last 4 s versus the target (sustained over-mapping);
  - a weak, capped cost for rests far longer than expected in dense music;
  - a soft stream limit (default 16 objects) when no hard `max_stream` is set.
- `ml/generation.py` calibration: the stars condition stays within ±1.5 ★ of the target and
  the trained range; an aim deficit lowers `tempo_bias` and asks for wider jumps
  (`jump_spacing` from the star profile, +15 % per missing star); only a speed deficit makes
  the rhythm denser; `tempo_bias` is limited to ±1.
- `ml/spinners.py`: spinners before loudness jumps (≥ 5 dB, about one bar, ending with at
  least one beat / half the approach time of recovery), in long audible gaps (≥ 8 beats) and
  in an audible outro. At most one drop spinner per 30 s; silence stays a break. The rhythm
  sampler treats a planned spinner as reserved time and resumes at the drop. Spinners are
  exported, re-parsed and checked for overlaps; the object after a spinner starts a combo.
- New request fields: `density_guidance` (0 disables, default 1), `dynamics` (0–2, default 1),
  `stream_soft_limit` (default 16, 0 disables), `spinners` (default true).
- Free validation reports `object_ratio` (generated / reference objects) and
  `longest_short_run`.

## Next steps for training

1. **Rhythm exposure bias.** Extend context robustness to the rhythm history: with a
   scheduled probability, replace the previous event's delta/gap features with the model's
   own sampled gap class during training, so a wrong gap is not simply repeated.
2. **Checkpoint selection on free generation.** Use `object_ratio` (|log2| close to 0) and
   `longest_short_run` relative to the reference next to the validation loss; the best
   teacher-forced loss was not the best free-running model.
3. **Learn spinners.** Add a spinner/break object class (or a "spinner" gap target) instead
   of dropping `Beatmap.ignored`, then replace the rule-based placement.
4. **Free validation with detected timing** on one song per run, so tempo-octave errors and
   calibration show up before a user generates a map.
5. Continue training at the reduced learning rate (5e-4) only together with 1–2; with the
   current objective additional epochs mostly improve teacher-forced metrics.

## Second run (after the changes above): on beat, but stacked and without combos

The tempo (198.8 BPM), the dynamics and the spinners worked. The map was still poor: median
spacing 5 px (jump spacing 2 px), the first half nearly only one-beat sliders, the second half
40 s of stacked 1/2 and 1/4 circles, 51 new combos for 530 objects (none after 96 s).

Causes and changes:

- **Stacking from the movement score.** Candidates were ranked against the model's expected
  movement features, a regression mean. For an uncertain direction it predicts cosine/sine
  close to 0, which is exactly a stacked note's value, and distances in /512 units made a
  150 px jump versus a stack almost free. Once stacked, the ratio to the previous distance
  kept it stacked. `candidate_selection.movement_cost` now weights direction by the model's
  certainty and compares the spacing on a log scale.
- **An almost empty style condition.** With only stars/AR/CS/OD/HP set, the model sees a
  condition it rarely trained with (free validation always passes the full reference style
  and moved 100–150 px). Unset style groups are now filled with the typical values of the
  target stars (`style_from_stars`, default on; groups the user touches stay the user's).
  The calibration round with a set jump spacing had already jumped from 2.6 to 5.2 stars.
- **Timing confidence out of distribution.** The audio fallback marked its beat grid with
  confidence 0.35; training only saw 1, 0.75 or no timing. Pattern mode now uses 0.75.
- **Combos.** The combo head's positive rate is about 1/8, so `logit > 0` almost never fires.
  `ml/combos.py` starts combos on bars (every bar when dense, every second bar otherwise),
  after rests of two beats or more and after spinners (`combos`, default on).
