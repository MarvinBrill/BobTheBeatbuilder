import { useId } from "react";
import { ChevronDown, RotateCcw, Star, UserRound } from "lucide-react";
import { HelpTooltip } from "../components/HelpTooltip";
import type { Model } from "../types";
import {
  formatStyleValue,
  starPresets,
  styleControls,
  type StyleControl,
} from "./styleControls";

export interface StyleTarget {
  useTarget: boolean;
  stars: number;
  /** Aim share of aim + speed difficulty; null lets the model decide. */
  aimShare: number | null;
  mapper: string;
  /** Explicit style values in UI units; missing keys are inferred by the model. */
  style: Record<string, number>;
  calibrate: boolean;
  rounds: number;
}

export const defaultTarget: StyleTarget = {
  useTarget: true,
  stars: 5,
  aimShare: null,
  mapper: "",
  style: {},
  calibrate: true,
  rounds: 3,
};

function clampTo(control: StyleControl, value: number) {
  return Math.min(control.max, Math.max(control.min, value));
}

function ValueSlider({
  label,
  help,
  value,
  shown,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  help: string;
  value: number;
  shown: string;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  const id = useId();
  return (
    <div className="slider-control">
      <div className="slider-heading">
        <label htmlFor={id}>{label}</label>
        <b>{shown}</b>
        <HelpTooltip label={label} text={help} />
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

/** Target stars, mapper and pattern controls for style-v2 models (or only stars for older ones). */
export function StudioStyle({
  model,
  value,
  onChange,
  onStars,
  calculatorReady,
  typical = {},
  legacy = false,
}: {
  model?: Model;
  /** Typical style values of the training maps near the target stars (UI units). */
  typical?: Record<string, number>;
  value: StyleTarget;
  onChange: (value: StyleTarget) => void;
  /** Called when a preset or the "typical values" button picks a difficulty. */
  onStars: (stars: number) => void;
  calculatorReady: boolean;
  legacy?: boolean;
}) {
  const update = (change: Partial<StyleTarget>) => onChange({ ...value, ...change });
  const mappers = Object.entries(model?.mapper_profiles || {}).sort(
    (left, right) => right[1].maps - left[1].maps || left[0].localeCompare(right[0]),
  );
  const explicit = Object.keys(value.style).length;

  function chooseMapper(name: string) {
    const profile = model?.mapper_profiles?.[name];
    if (!profile) return update({ mapper: "" });
    const style: Record<string, number> = {};
    for (const control of styleControls) {
      const measured = profile.style[control.key];
      // Density depends on the song and the target stars more than on the mapper.
      if (measured !== undefined && control.key !== "density")
        style[control.key] = clampTo(control, measured);
    }
    update({ mapper: name, style });
  }

  return (
    <div className="style-target">
      {!legacy && (
      <div className="preset-row five" role="group" aria-label="Ziel-Schwierigkeit">
        {starPresets.map((preset) => (
          <button
            type="button"
            key={preset.name}
            aria-pressed={value.useTarget && Math.abs(value.stars - preset.stars) < 0.05}
            onClick={() => {
              update({ useTarget: true, stars: preset.stars });
              onStars(preset.stars);
            }}
          >
            {preset.name}
            <small>{preset.stars.toFixed(1)} ★</small>
          </button>
        ))}
      </div>
      )}
      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={value.useTarget}
          onChange={(event) => update({ useTarget: event.target.checked })}
        />{" "}
        <Star size={15} aria-hidden="true" />{" "}
        {legacy ? "Auf Ziel-Sterne kalibrieren (Dichte und Abstand werden nachgeregelt)" : "Ziel-Sterne vorgeben"}
      </label>
      {value.useTarget && (
        <>
          <ValueSlider
            label="Ziel-Sterne"
            help="Star Rating, wie osu! es berechnet (Aim, Speed und Reading). Das Modell hat gelernt, wie Maps dieser Schwierigkeit aussehen. Mit Kalibrierung wird die fertige Map nachgemessen und bei Abweichung neu erzeugt."
            value={value.stars}
            shown={`${value.stars.toFixed(1)} ★`}
            min={1}
            max={10}
            step={0.1}
            onChange={(stars) => update({ stars })}
          />
          <button type="button" className="text-button inline" onClick={() => onStars(value.stars)}>
            AR/OD/HP/CS für {value.stars.toFixed(1)} ★ aus deinen Trainingsdaten übernehmen
          </button>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={value.calibrate && calculatorReady}
              disabled={!calculatorReady}
              onChange={(event) => update({ calibrate: event.target.checked })}
            />{" "}
            Ergebnis nachmessen und bis zu {value.rounds}× nachregeln
          </label>
          {!calculatorReady && (
            <p className="fine-print">
              Der Sterne-Rechner fehlt. Führe scripts/setup.ps1 erneut aus, dann werden fertige Maps
              mit der offiziellen osu!-Berechnung gemessen.
            </p>
          )}
        </>
      )}
      {!legacy && (
        <>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={value.aimShare !== null}
              onChange={(event) => update({ aimShare: event.target.checked ? 0.55 : null })}
            />{" "}
            Aim/Speed-Verteilung festlegen
          </label>
          {value.aimShare !== null && (
            <ValueSlider
              label="Aim-Anteil"
              help="Anteil der Aim-Schwierigkeit an Aim + Speed. Höher: Sprung-lastige Map (Snap/Flow-Aim). Niedriger: Tapping-lastige Map mit Bursts und Streams, die alternierend gespielt werden."
              value={value.aimShare}
              shown={`${Math.round(value.aimShare * 100)} % Aim · ${Math.round((1 - value.aimShare) * 100)} % Speed`}
              min={0.3}
              max={0.8}
              step={0.01}
              onChange={(aimShare) => update({ aimShare })}
            />
          )}
          <label>
            <span className="label-with-icon">
              <UserRound size={15} aria-hidden="true" /> Mapper-Stil
            </span>
            <select value={value.mapper} onChange={(event) => chooseMapper(event.target.value)}>
              <option value="">Kein bestimmter Mapper</option>
              {mappers.map(([name, profile]) => (
                <option key={name} value={name}>
                  {name} · {profile.maps} Maps
                  {profile.stars_min != null && profile.stars_max != null
                    ? ` · ${profile.stars_min.toFixed(1)}–${profile.stars_max.toFixed(1)} ★`
                    : ""}
                </option>
              ))}
            </select>
            <small>
              {mappers.length
                ? "Nutzt das gelernte Mapper-Profil und übernimmt seine typischen Pattern-Werte. Gast-Difficulties zählen für den Gast-Mapper."
                : "Dieses Modell kennt noch keine Mapper (mindestens 10 Trainingsmaps pro Mapper nötig)."}
            </small>
          </label>
          <details className="advanced pattern-controls" open={explicit > 0}>
            <summary>
              Patterns & Bewegung
              {explicit ? ` · ${explicit} festgelegt` : " · automatisch"}
              <ChevronDown size={16} />
            </summary>
            <div className="advanced-content">
              <p className="fine-print">
                Nicht festgelegte Werte leitet das Modell aus Musik, Ziel-Sternen und Mapper ab.
              </p>
              {explicit > 0 && (
                <button type="button" className="text-button inline" onClick={() => update({ style: {} })}>
                  <RotateCcw size={14} aria-hidden="true" /> Alle auf automatisch
                </button>
              )}
              {(["Rhythmus", "Bewegung", "Slider", "Struktur"] as const).map((group) => (
                <fieldset key={group} className="pattern-group">
                  <legend>{group}</legend>
                  {styleControls
                    .filter((control) => control.group === group)
                    .map((control) => {
                      const chosen = value.style[control.key];
                      return (
                        <div key={control.key} className="pattern-control">
                          <label className="checkbox-label">
                            <input
                              type="checkbox"
                              checked={chosen !== undefined}
                              onChange={(event) => {
                                const style = { ...value.style };
                                if (event.target.checked)
                                  style[control.key] = clampTo(
                                    control,
                                    typical[control.key] ?? (control.min + control.max) / 2,
                                  );
                                else delete style[control.key];
                                update({ style });
                              }}
                            />{" "}
                            {control.label}
                            {chosen === undefined && <small> · automatisch</small>}
                          </label>
                          {chosen !== undefined && (
                            <ValueSlider
                              label={control.label}
                              help={control.help}
                              value={chosen}
                              shown={formatStyleValue(control, chosen)}
                              min={control.min}
                              max={control.max}
                              step={control.step}
                              onChange={(next) =>
                                update({ style: { ...value.style, [control.key]: next } })
                              }
                            />
                          )}
                        </div>
                      );
                    })}
                </fieldset>
              ))}
            </div>
          </details>
        </>
      )}
    </div>
  );
}
