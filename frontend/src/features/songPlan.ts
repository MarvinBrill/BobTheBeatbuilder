import type { PlanSection, PlanStyle, PlanTimingPoint, SectionKind, SongPlan } from "../types";

export const kindLabels: Record<SectionKind, string> = {
  intro: "Intro",
  verse: "Strophe",
  build: "Build-up",
  drop: "Drop/Refrain",
  break: "Break",
  outro: "Outro",
  silence: "Stille",
};

export const styleLabels: Record<PlanStyle, string> = {
  auto: "Automatisch",
  calm: "Ruhig",
  flow: "Flow",
  jumps: "Jumps",
  streams: "Streams",
  mixed: "Gemischt",
};

/** The style the backend uses for "auto" (ml/song_plan.py DEFAULT_STYLE). */
export const defaultStyle: Record<SectionKind, PlanStyle> = {
  intro: "calm",
  verse: "flow",
  build: "streams",
  drop: "jumps",
  break: "calm",
  outro: "calm",
  silence: "calm",
};

export function bpmOf(point?: PlanTimingPoint) {
  return point ? 60000 / point.beat_length : 0;
}

export function clock(ms: number) {
  const total = Math.max(0, Math.round(ms));
  const minutes = Math.floor(total / 60000);
  const seconds = Math.floor(total / 1000) % 60;
  const millis = total % 1000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

export function pointAt(plan: SongPlan, time: number) {
  let current = plan.timing[0];
  for (const point of plan.timing) if (point.time <= time) current = point;
  return current;
}

export function barLength(plan: SongPlan, time: number) {
  const point = pointAt(plan, time);
  return point ? point.beat_length * (point.meter || 4) : 2000;
}

/** Every beat in [start, end) with whether it starts a bar. */
export function beatsBetween(plan: SongPlan, start: number, end: number) {
  const beats: { time: number; downbeat: boolean }[] = [];
  plan.timing.forEach((point, index) => {
    const next = plan.timing[index + 1]?.time ?? Infinity;
    const meter = point.meter || 4;
    const origin = point.bar_origin ?? point.time;
    const from = Math.max(start, point.time);
    const to = Math.min(end, next);
    if (to <= from) return;
    let k = Math.ceil((from - point.time) / point.beat_length - 1e-9);
    for (let time = point.time + k * point.beat_length; time < to; k++, time = point.time + k * point.beat_length) {
      const inBar = Math.round((time - origin) / point.beat_length);
      beats.push({ time, downbeat: ((inBar % meter) + meter) % meter === 0 });
    }
  });
  return beats;
}

/** Shift every red line (and bar origin) by `ms`; the sections keep their times. */
export function shiftTiming(plan: SongPlan, ms: number): SongPlan {
  return {
    ...plan,
    timing_source: "edited",
    timing: plan.timing.map((point) => ({
      ...point,
      time: point.time + ms,
      bar_origin: point.bar_origin == null ? point.bar_origin : point.bar_origin + ms,
    })),
  };
}

/** A new tempo for the red line at `index`, keeping its first beat where it is. */
export function setTempo(plan: SongPlan, index: number, bpm: number): SongPlan {
  if (!(bpm >= 20 && bpm <= 400)) return plan;
  return {
    ...plan,
    timing_source: "edited",
    timing: plan.timing.map((point, position) =>
      position === index ? { ...point, beat_length: 60000 / bpm } : point,
    ),
  };
}

export function updateSection(plan: SongPlan, index: number, patch: Partial<PlanSection>): SongPlan {
  return {
    ...plan,
    sections: plan.sections.map((section, position) =>
      position === index ? { ...section, ...patch } : section,
    ),
  };
}

/** Move the start of section `index` by whole bars (the previous section follows). */
export function moveBoundary(plan: SongPlan, index: number, bars: number): SongPlan {
  if (index <= 0 || index >= plan.sections.length) return plan;
  const sections = plan.sections.map((section) => ({ ...section }));
  const current = sections[index];
  const previous = sections[index - 1];
  const step = barLength(plan, current.start_ms) * bars;
  const start = current.start_ms + step;
  if (start <= previous.start_ms + 1 || start >= current.end_ms - 1) return plan;
  current.start_ms = start;
  previous.end_ms = start;
  return { ...plan, sections };
}

export function mergeWithNext(plan: SongPlan, index: number): SongPlan {
  if (index < 0 || index + 1 >= plan.sections.length) return plan;
  const sections = plan.sections.map((section) => ({ ...section }));
  sections[index].end_ms = sections[index + 1].end_ms;
  sections[index].bars = (sections[index].bars || 0) + (sections[index + 1].bars || 0);
  sections.splice(index + 1, 1);
  return { ...plan, sections: renumberRepeats(sections, index + 1) };
}

/** Split a section on the bar line nearest to its middle. */
export function splitSection(plan: SongPlan, index: number): SongPlan {
  const section = plan.sections[index];
  if (!section) return plan;
  const bar = barLength(plan, section.start_ms);
  const bars = Math.round((section.end_ms - section.start_ms) / bar);
  if (bars < 2) return plan;
  const middle = section.start_ms + Math.floor(bars / 2) * bar;
  const sections = plan.sections.map((item) => ({ ...item }));
  const second = { ...section, start_ms: middle, bars: bars - Math.floor(bars / 2), repeat_of: null };
  sections[index] = { ...section, end_ms: middle, bars: Math.floor(bars / 2) };
  sections.splice(index + 1, 0, second);
  return { ...plan, sections: renumberRepeats(sections, index + 1, 1) };
}

function renumberRepeats(sections: PlanSection[], from: number, delta = -1) {
  return sections.map((section) => {
    const repeat = section.repeat_of;
    if (repeat == null) return section;
    if (delta < 0 && repeat === from) return { ...section, repeat_of: null };
    return repeat >= from ? { ...section, repeat_of: repeat + delta } : section;
  });
}
