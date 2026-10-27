import type { Device, Hardware } from "../types";
import { useId } from "react";
import { HelpTooltip } from "./HelpTooltip";

export function Slider({
  label,
  value,
  onChange,
  min = 0,
  max = 10,
  step = 0.5,
  hint,
  help,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
  help: string;
}) {
  const id = useId();
  return (
    <div className="slider-control">
      <div className="slider-heading">
        <label htmlFor={id}>{label}</label>
        <b>{value.toFixed(String(step).split(".")[1]?.length || 0)}</b>
        <HelpTooltip label={label} text={help} />
      </div>
      <input
        id={id}
        aria-describedby={`${id}-help`}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span id={`${id}-help`} className="visually-hidden">
        {help}
      </span>
      {hint && <small>{hint}</small>}
    </div>
  );
}

export function DeviceSelect({
  value,
  onChange,
  hardware,
}: {
  value: Device;
  onChange: (device: Device) => void;
  hardware: Hardware | null;
}) {
  return (
    <label>
      Rechengerät
      <select
        value={value}
        onChange={(event) => onChange(event.target.value as Device)}
      >
        <option value="cpu">CPU</option>
        <option value="cuda" disabled={!hardware?.cuda}>
          NVIDIA CUDA{!hardware?.cuda ? " · nicht verfügbar" : ""}
        </option>
        <option value="directml" disabled={!hardware?.directml}>
          DirectML{!hardware?.directml ? " · nicht verfügbar" : ""}
        </option>
      </select>
    </label>
  );
}
