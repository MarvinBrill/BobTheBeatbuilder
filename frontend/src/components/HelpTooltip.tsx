import { useEffect, useId, useState } from "react";
import { createPortal } from "react-dom";
import { CircleHelp } from "lucide-react";

export function HelpTooltip({ label, text }: { label: string; text: string }) {
  const id = useId();
  const [position, setPosition] = useState<{
    left: number;
    top?: number;
    bottom?: number;
    width: number;
  } | null>(null);
  useEffect(() => {
    const close = () => setPosition(null);
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, []);
  function show(button: HTMLButtonElement) {
    const rect = button.getBoundingClientRect();
    const width = Math.min(300, window.innerWidth - 24);
    setPosition({
      left: Math.max(
        12,
        Math.min(rect.right - width, window.innerWidth - width - 12),
      ),
      width,
      ...(rect.bottom > window.innerHeight - 180
        ? { bottom: window.innerHeight - rect.top + 8 }
        : { top: rect.bottom + 8 }),
    });
  }
  return (
    <>
      <button
        type="button"
        className="help-tooltip-button"
        aria-label={`Hilfe zu ${label}`}
        aria-describedby={position ? id : undefined}
        onMouseEnter={(event) => show(event.currentTarget)}
        onMouseLeave={(event) => {
          if (document.activeElement !== event.currentTarget) setPosition(null);
        }}
        onFocus={(event) => show(event.currentTarget)}
        onBlur={() => setPosition(null)}
        onClick={(event) => show(event.currentTarget)}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            setPosition(null);
            event.stopPropagation();
          }
        }}
      >
        <CircleHelp size={16} aria-hidden="true" />
      </button>
      {position &&
        createPortal(
          <div id={id} role="tooltip" className="help-tooltip" style={position}>
            {text}
          </div>,
          document.body,
        )}
    </>
  );
}
