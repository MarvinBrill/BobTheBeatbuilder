import { Volume2, VolumeX } from "lucide-react";
import { useEffect, useState } from "react";

export function usePreviewVolume(key: string) {
  const [volume, setVolume] = useState(() => {
    try {
      const saved = localStorage.getItem(key);
      const value = saved === null ? 0.5 : Number(saved);
      return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0.5;
    } catch {
      return 0.5;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, String(volume));
    } catch {
      /* Optional preference. */
    }
  }, [key, volume]);
  return [volume, setVolume] as const;
}

export function PreviewVolume({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="preview-volume">
      {value === 0 ? <VolumeX size={18} /> : <Volume2 size={18} />}
      <span>{label}</span>
      <input
        type="range"
        aria-label={`${label}-Lautstärke`}
        min="0"
        max="1"
        step="0.01"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <output>{Math.round(value * 100)} %</output>
    </label>
  );
}
