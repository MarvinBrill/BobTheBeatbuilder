import { useState, type DragEvent, type ReactNode } from "react";
import { LoaderCircle, type LucideIcon } from "lucide-react";

/** The button's own icon, swapped for a spinner while its request runs. */
export function BusyIcon({
  busy,
  icon: Icon,
  size = 17,
}: {
  busy: boolean;
  icon: LucideIcon;
  size?: number;
}) {
  return busy ? (
    <LoaderCircle size={size} className="spin" aria-hidden="true" />
  ) : (
    <Icon size={size} aria-hidden="true" />
  );
}

/** Says why the control beside it is disabled, or what it is doing. */
export function ActionHint({
  id,
  children,
  tone = "neutral",
}: {
  id?: string;
  children?: ReactNode;
  tone?: "neutral" | "error";
}) {
  if (!children) return null;
  return (
    <p
      id={id}
      className={`action-hint ${tone === "error" ? "error" : ""}`}
      role={tone === "error" ? "alert" : undefined}
    >
      {children}
    </p>
  );
}

/** A determinate bar with its percentage, for uploads and running jobs. */
export function ProgressLine({
  value,
  label,
}: {
  value: number;
  label: string;
}) {
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="progress-line">
      <span>
        {label} <b>{percent} %</b>
      </span>
      <progress max={1} value={value} aria-label={label} />
    </div>
  );
}

/** Drag-over state for a dropzone; hands the first dropped file on. */
export function useDropTarget(
  onFile: (file: File) => void,
  disabled = false,
) {
  const [dragging, setDragging] = useState(false);
  return {
    dragging,
    handlers: {
      onDragOver: (event: DragEvent) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      },
      onDragLeave: (event: DragEvent) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null))
          setDragging(false);
      },
      onDrop: (event: DragEvent) => {
        event.preventDefault();
        setDragging(false);
        const file = event.dataTransfer.files[0];
        if (file && !disabled) onFile(file);
      },
    },
  };
}
