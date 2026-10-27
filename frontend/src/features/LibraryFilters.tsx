import { Layers, RotateCcw, SlidersHorizontal, Star } from "lucide-react";
import type { MapFilters } from "./useLibraryMaps";

const songOptions = [
  ["all", "Alle Maps"],
  ["easiest", "Leichteste"],
  ["middle", "Mittlere"],
  ["hardest", "Schwerste"],
] as const;

export function LibraryFilters({
  filters,
  onChange,
  invalid,
}: {
  filters: MapFilters;
  onChange: (change: Partial<MapFilters>) => void;
  invalid: boolean;
}) {
  return (
    <div className="library-filters">
      <div className="filter-heading">
        <span>
          <SlidersHorizontal size={17} /> Deine Trainingsauswahl
        </span>
        <button
          className="text-button"
          onClick={() => onChange({ minimum: "", maximum: "", perSong: "all" })}
        >
          <RotateCcw size={13} /> Zurücksetzen
        </button>
      </div>
      <div className="filter-fields">
        <div className="filter-block">
          <div className="filter-label">
            <Star size={15} /> Sternebereich
          </div>
          <div className="star-range">
            <label>
              <span>Von</span>
              <input
                aria-label="Sterne von"
                type="number"
                min="0"
                step="0.1"
                placeholder="Offen"
                value={filters.minimum}
                aria-invalid={invalid}
                onChange={(event) => onChange({ minimum: event.target.value })}
              />
            </label>
            <span className="range-separator" aria-hidden="true">
              —
            </span>
            <label>
              <span>Bis</span>
              <input
                aria-label="Sterne bis"
                type="number"
                min="0"
                step="0.1"
                placeholder="Offen"
                value={filters.maximum}
                aria-invalid={invalid}
                onChange={(event) => onChange({ maximum: event.target.value })}
              />
            </label>
          </div>
        </div>
        <div className="filter-block">
          <div className="filter-label">
            <Layers size={15} /> Schwierigkeiten pro Song
          </div>
          <div
            className="song-filter-options"
            role="group"
            aria-label="Schwierigkeiten pro Song"
          >
            {songOptions.map(([value, label]) => (
              <button
                key={value}
                aria-pressed={filters.perSong === value}
                onClick={() => onChange({ perSong: value })}
              >
                {label}
              </button>
            ))}
          </div>
          <small>
            {filters.perSong === "all"
              ? "Alle passenden Schwierigkeiten anzeigen."
              : "Genau eine passende Map je Song – über alle Seiten."}
          </small>
        </div>
      </div>
      <p className="filter-explanation" role={invalid ? "alert" : undefined}>
        {invalid ? (
          "Sterne von muss kleiner oder gleich Sterne bis sein; nur nicht negative Zahlen sind erlaubt."
        ) : (
          <>
            Zuerst Suche und Sternebereich, danach die Schwierigkeit je Song.
            Beide Sternegrenzen zählen mit.
            {filters.perSong === "middle" &&
              " Bei gerader Anzahl wird die leichtere der beiden mittleren Maps gewählt."}
            {filters.perSong !== "all" || filters.minimum || filters.maximum
              ? " Maps ohne Sternebewertung bleiben ausgeblendet."
              : " Sternebewertungen stammen aus lazer."}{" "}
            Filteränderungen leeren die Auswahl.
          </>
        )}
      </p>
    </div>
  );
}
