import { ChevronDown } from "lucide-react";
import { DeviceSelect, Slider } from "../components/Controls";
import type { Device, Hardware } from "../types";
import { generationHelp } from "./generationHelp";

export const defaultAdvanced = {
  bpm: "",
  offset: 0,
  snap: true,
  divisor: 4,
  seed: 42,
  temperature: 0.8,
  candidates: 12,
  max_stream: 0,
  rhythm_temperature: 0.6,
  rhythm_mode: "patterns" as "patterns" | "onsets",
};

export function StudioAdvanced({
  value,
  onChange,
  device,
  setDevice,
  hardware,
  version,
  setVersion,
  styled = false,
}: {
  styled?: boolean;
  value: typeof defaultAdvanced;
  onChange: (value: typeof defaultAdvanced) => void;
  device: Device;
  setDevice: (device: Device) => void;
  hardware: Hardware | null;
  version: string;
  setVersion: (value: string) => void;
}) {
  const update = (change: Partial<typeof defaultAdvanced>) =>
    onChange({ ...value, ...change });
  return (
    <details className="advanced">
      <summary>
        Feineinstellungen <ChevronDown size={16} />
      </summary>
      <div className="advanced-content">
        <div className="field-row">
          <label>
            BPM
            <input
              type="number"
              min={20}
              max={400}
              value={value.bpm}
              placeholder="Automatisch"
              onChange={(event) => update({ bpm: event.target.value })}
            />
          </label>
          <label>
            Offset (ms, bei manuellen BPM)
            <input
              type="number"
              min={-10000}
              max={60000}
              value={value.offset}
              onChange={(event) =>
                update({ offset: Number(event.target.value) })
              }
            />
          </label>
        </div>
        <div className="field-row">
          <label>
            Beat-Raster
            <select
              value={value.divisor}
              onChange={(event) =>
                update({ divisor: Number(event.target.value) })
              }
            >
              {[1, 2, 3, 4, 6, 8, 12].map((number) => (
                <option key={number} value={number}>
                  {number === 12 ? "1/12 (1/4 und 1/3 gemischt)" : `1/${number}`}
                </option>
              ))}
            </select>
          </label>
          <label>
            Difficulty-Name
            <input
              value={version}
              onChange={(event) => setVersion(event.target.value)}
            />
          </label>
        </div>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={value.snap}
            onChange={(event) => update({ snap: event.target.checked })}
          />{" "}
          Nahe Treffer am Raster ausrichten
        </label>
        {styled && (
          <label>
            Rhythmus-Auswahl
            <select
              value={value.rhythm_mode}
              onChange={(event) =>
                update({
                  rhythm_mode: event.target.value as "patterns" | "onsets",
                })
              }
            >
              <option value="patterns">Patterns: Modell wählt jede nächste Note</option>
              <option value="onsets">Onsets: erst alle Zeitpunkte, dann Objekte</option>
            </select>
            <small>{generationHelp.rhythmMode}</small>
          </label>
        )}
        {styled && value.rhythm_mode === "patterns" && (
          <div className="field-row">
            <label>
              Längster Stream
              <input
                type="number"
                min={0}
                max={256}
                value={value.max_stream}
                onChange={(event) => update({ max_stream: Number(event.target.value) })}
              />
              <small>Obergrenze für zusammenhängende 1/4-Noten. 0 = keine Grenze.</small>
            </label>
            <Slider
              label="Rhythmus-Vielfalt"
              help={generationHelp.rhythm}
              value={value.rhythm_temperature}
              min={0.1}
              max={1.5}
              step={0.05}
              onChange={(rhythm_temperature) => update({ rhythm_temperature })}
            />
          </div>
        )}
        <Slider
          label="Variantenvielfalt"
          help={generationHelp.temperature}
          value={value.temperature}
          min={0.1}
          max={1.5}
          step={0.05}
          onChange={(temperature) => update({ temperature })}
        />
        <div className="field-row">
          <label>
            Kandidaten pro Objekt
            <input
              type="number"
              min={1}
              max={48}
              value={value.candidates}
              onChange={(event) =>
                update({ candidates: Number(event.target.value) })
              }
            />
            <small>
              Mehr Kandidaten erlauben eine breitere Auswahl, brauchen aber
              länger.
            </small>
          </label>
          <label>
            Varianten-Seed
            <input
              type="number"
              min={0}
              max={2147483647}
              value={value.seed}
              onChange={(event) => update({ seed: Number(event.target.value) })}
            />
            <small>
              Gleicher Seed für wiederholbare Varianten auf derselben Hardware.
            </small>
          </label>
        </div>
        <DeviceSelect value={device} onChange={setDevice} hardware={hardware} />
      </div>
    </details>
  );
}
