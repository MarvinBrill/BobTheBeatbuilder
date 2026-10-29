import type { Model, StyleProfile } from "../types";

/** One measurable style value. Keys match backend/app/ml/style.py. */
export interface StyleControl {
  key: string;
  label: string;
  group: "Rhythmus" | "Bewegung" | "Slider" | "Struktur";
  min: number;
  max: number;
  step: number;
  /** Shown as percentage of the value (0..1 values). */
  percent?: boolean;
  unit?: string;
  help: string;
}

export const styleControls: StyleControl[] = [
  {
    key: "density",
    label: "Objektdichte",
    group: "Rhythmus",
    min: 0.5,
    max: 10,
    step: 0.1,
    unit: "/s",
    help: "Objektstarts pro Sekunde über die ganze Map, Pausen eingerechnet. Ein Slider zählt als ein Objekt.",
  },
  {
    key: "burst_share",
    label: "Bursts (3–8 Noten)",
    group: "Rhythmus",
    min: 0,
    max: 0.6,
    step: 0.01,
    percent: true,
    help: "Anteil der Objekte in kurzen 1/4-Folgen: Triples, 5er-Bursts. Bei hohem BPM nur alternierend spielbar.",
  },
  {
    key: "stream_share",
    label: "Streams (9+ Noten)",
    group: "Rhythmus",
    min: 0,
    max: 0.7,
    step: 0.01,
    percent: true,
    help: "Anteil der Objekte in langen 1/4-Folgen. Viel Stream-Anteil erhöht vor allem die Speed-Sterne.",
  },
  {
    key: "stream_length",
    label: "Ø Länge Bursts/Streams",
    group: "Rhythmus",
    min: 3,
    max: 48,
    step: 1,
    unit: " Noten",
    help: "Durchschnittliche Länge aller 1/4-Folgen ab drei Noten. Unter „Feineinstellungen“ lässt sich zusätzlich eine harte Obergrenze setzen.",
  },
  {
    key: "triplet_share",
    label: "1/3- und 1/6-Rhythmen",
    group: "Rhythmus",
    min: 0,
    max: 0.4,
    step: 0.01,
    percent: true,
    help: "Anteil der Abstände im Triolen-Raster. Braucht das Beat-Raster 1/12 (gemischt) oder 1/6.",
  },
  {
    key: "jump_spacing",
    label: "Sprungweite bei 1/2",
    group: "Bewegung",
    min: 20,
    max: 400,
    step: 5,
    unit: " px",
    help: "Typischer Abstand zwischen Noten im 1/2-Abstand (Median, Spielfeld 512×384 px). Weiter = mehr Aim-Sterne.",
  },
  {
    key: "stream_spacing",
    label: "Abstand in Streams",
    group: "Bewegung",
    min: 0,
    max: 140,
    step: 1,
    unit: " px",
    help: "0–10 px: gestapelte Streams. 20–40 px: normale Streams. Darüber: Spaced Streams.",
  },
  {
    key: "flow",
    label: "Richtungsfluss",
    group: "Bewegung",
    min: 0,
    max: 1,
    step: 0.01,
    help: "Niedrig: viele Richtungswechsel (Hin-und-her-Sprünge). Hoch: fließende, gleichgerichtete Bewegung.",
  },
  {
    key: "sharp_angles",
    label: "Spitze Winkel",
    group: "Bewegung",
    min: 0,
    max: 1,
    step: 0.01,
    percent: true,
    help: "Anteil der Sprünge mit Winkel unter 60°: Back-and-forth, Zickzack, Dreiecke mit Snap-Aim.",
  },
  {
    key: "linear_angles",
    label: "Gerade Linien",
    group: "Bewegung",
    min: 0,
    max: 1,
    step: 0.01,
    percent: true,
    help: "Anteil der Sprünge mit Winkel über 120°: lineare Jumps, Flow-Aim.",
  },
  {
    key: "stacks",
    label: "Stacks",
    group: "Bewegung",
    min: 0,
    max: 0.5,
    step: 0.01,
    percent: true,
    help: "Anteil der direkt übereinander liegenden Noten, zum Beispiel bei Doubles oder gestapelten Bursts.",
  },
  {
    key: "slider_ratio",
    label: "Slider-Anteil",
    group: "Slider",
    min: 0,
    max: 1,
    step: 0.01,
    percent: true,
    help: "Anteil der Objekte, die Slider sind. 0 erzeugt nur Circles.",
  },
  {
    key: "slider_velocity",
    label: "Slider-Tempo",
    group: "Slider",
    min: 50,
    max: 700,
    step: 10,
    unit: " px/Beat",
    help: "Wie weit ein Slider pro Beat läuft. 140 px/Beat entsprechen Slider Multiplier 1,4 ohne grüne Linien.",
  },
  {
    key: "short_sliders",
    label: "Kurze Slider (≤ 1/2)",
    group: "Slider",
    min: 0,
    max: 1,
    step: 0.01,
    percent: true,
    help: "Anteil kurzer Slider, etwa Kick-Slider auf Bass-Schlägen.",
  },
  {
    key: "repeat_sliders",
    label: "Wiederholungs-Slider",
    group: "Slider",
    min: 0,
    max: 0.6,
    step: 0.01,
    percent: true,
    help: "Anteil der Slider mit Repeats (Reverse-Arrows).",
  },
  {
    key: "combo_length",
    label: "Combo-Länge",
    group: "Struktur",
    min: 2,
    max: 16,
    step: 0.5,
    unit: " Objekte",
    help: "Durchschnittliche Objekte pro New Combo. Viele Mapper setzen New Combos auf Takt- oder Phrasenanfänge.",
  },
];

export const styleControlKeys = new Set(styleControls.map((control) => control.key));

export function formatStyleValue(control: StyleControl, value: number) {
  if (control.percent) return `${Math.round(value * 100)} %`;
  const digits = control.step >= 1 ? 0 : String(control.step).split(".")[1]?.length || 1;
  return `${value.toFixed(digits)}${control.unit || ""}`;
}

export function isStyled(model?: Model) {
  return (model?.style_version ?? Number(model?.model_config?.style_version ?? 1)) >= 2;
}

/** Typical values in this model's training data for the nearest star bucket. */
export function starProfile(model: Model | undefined, stars: number): StyleProfile | undefined {
  const profiles = model?.star_profiles;
  if (!profiles) return undefined;
  let best: [number, StyleProfile] | undefined;
  for (const [bucket, profile] of Object.entries(profiles)) {
    const distance = Math.abs(Number(bucket) - stars);
    if (!best || distance < best[0]) best = [distance, profile];
  }
  return best && best[0] <= 1 ? best[1] : undefined;
}

/** Fallback when a model has no star profiles: common values of ranked maps. */
export function typicalSettings(stars: number) {
  const clamp = (value: number, low: number, high: number) =>
    Math.round(Math.min(high, Math.max(low, value)) * 10) / 10;
  return {
    ar: clamp(4 + stars * 0.95, 3, 9.8),
    od: clamp(2 + stars * 1.15, 2, 9.5),
    hp: clamp(2 + stars * 0.7, 2, 7),
    cs: 4,
  };
}

export const starPresets = [
  { name: "Easy", stars: 1.8 },
  { name: "Normal", stars: 2.6 },
  { name: "Hard", stars: 3.6 },
  { name: "Insane", stars: 4.9 },
  { name: "Extra", stars: 6.0 },
];
