import { useState } from "react";
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  Database,
  Download,
  FolderOpen,
  LoaderCircle,
  RefreshCw,
  Search,
  Star,
  Upload,
  X,
} from "lucide-react";
import { acceptsFile, api, upload } from "../api";
import {
  ActionHint,
  BusyIcon,
  ProgressLine,
  useDropTarget,
} from "../components/Feedback";
import type { State } from "../types";
import { activeJob, jobMessage } from "./jobs";
import { usePendingJob } from "./usePendingJob";

import { useLibraryMaps } from "./useLibraryMaps";
import { LibraryFilters } from "./LibraryFilters";

interface Directory {
  path: string;
  parent: string;
  children: { name: string; path: string }[];
}

const ARCHIVE_TYPES = [".osz", ".zip"];

export function Library({
  state,
  refresh,
  onError,
}: {
  state: State;
  refresh: () => void;
  onError: (message: string) => void;
}) {
  const [path, setPath] = useState(state.default_path);
  const [directory, setDirectory] = useState<Directory | null>(null);
  const revision = `${state.map_count}:${state.jobs
    .filter((job) => ["scan", "import", "stars"].includes(job.kind))
    .map((job) => `${job.id}:${job.status}`)
    .join(",")}`;
  const {
    filters,
    updateFilters,
    invalid,
    offset,
    setOffset,
    items,
    total,
    selected,
    togglePage,
    loading,
    selecting,
    selectAll,
    toggle,
    clearSelection,
    error,
    retry,
  } = useLibraryMaps(revision, onError);
  const [name, setName] = useState("Meine Mapping-Library");
  /** Which request is in flight: one at a time, each with its own spinner. */
  const [request, setRequest] = useState<
    "" | "browse" | "scan" | "dataset" | "export" | "stars"
  >("");
  const [lastJob, setLastJob] = useState<
    "" | "scan" | "dataset" | "export" | "stars"
  >("");
  const pendingJob = usePendingJob(state.jobs);
  const busy = request || (pendingJob.waiting ? lastJob : "");
  const [uploading, setUploading] = useState<number | null>(null);
  const scanning = activeJob(state.jobs, "scan");
  const importing = activeJob(state.jobs, "import");
  const rating = activeJob(state.jobs, "stars");
  const lastRating = state.jobs.find((job) => job.kind === "stars");

  async function action(
    kind: "scan" | "dataset" | "export" | "stars",
    operation: () => Promise<unknown>,
  ) {
    setRequest(kind);
    try {
      const job = await operation();
      setLastJob(kind);
      pendingJob.track(job);
      refresh();
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setRequest("");
    }
  }

  async function browse(target: string) {
    setRequest("browse");
    try {
      const result = await api<Directory>(
        `/browse?path=${encodeURIComponent(target)}`,
      );
      setDirectory(result);
      setPath(result.path);
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setRequest("");
    }
  }

  async function importArchive(file?: File) {
    if (!file || uploading !== null) return;
    if (!acceptsFile(file, ARCHIVE_TYPES)) {
      onError(`„${file.name}“ ist kein .osz- oder .zip-Archiv.`);
      return;
    }
    setUploading(0);
    try {
      await upload("/import", file, setUploading);
      refresh();
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setUploading(null);
    }
  }
  const drop = useDropTarget(
    (file) => void importArchive(file),
    uploading !== null,
  );

  const filtered =
    !!filters.search ||
    !!filters.minimum ||
    !!filters.maximum ||
    filters.perSong !== "all";
  // A running scan shows its own progress line below the button.
  const scanReason = scanning
    ? ""
    : !path.trim()
      ? "Gib den osu!-Ordner an."
      : "";
  const datasetReason = !selected.size
    ? "Wähle zuerst Maps in der Tabelle aus."
    : !name.trim()
      ? "Gib dem Datensatz einen Namen."
      : "";
  const lastOffset = Math.min(offset + 100, total);

  return (
    <>
      <div className="page-heading">
        <div>
          <h1>
            Deine <span className="accent">Library.</span>
          </h1>
          <p>
            Gute Maps, gute Grundlage. Importiere deine osu!-Sammlung und
            kuratiere deinen Trainingsstil.
          </p>
        </div>
        <span className="pill">
          {state.map_count.toLocaleString("de")} Maps
        </span>
      </div>
      <div className="library-import-grid">
        <section className="panel">
          <div className="panel-heading">
            <h2>
              <FolderOpen size={18} aria-hidden="true" /> Aus osu! importieren
            </h2>
            <span className="tag">lazer & stable</span>
          </div>
          <label>
            osu!-Datenordner oder Songs-Verzeichnis
            <div className="input-action">
              <input
                value={path}
                onChange={(event) => setPath(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && path.trim() && !busy)
                    void browse(path);
                }}
              />
              <button
                type="button"
                className="secondary"
                disabled={!!busy || !path.trim()}
                aria-busy={busy === "browse"}
                onClick={() => void browse(path)}
              >
                <BusyIcon busy={busy === "browse"} icon={FolderOpen} size={16} />
                Durchsuchen
              </button>
            </div>
          </label>
          {directory && (
            <div
              className="directory-browser"
              aria-busy={busy === "browse"}
              aria-label="Ordner auswählen"
              role="group"
            >
              <div className="directory-head">
                <button
                  type="button"
                  disabled={!!busy}
                  onClick={() => void browse(directory.parent)}
                >
                  <ArrowLeft size={14} aria-hidden="true" /> Übergeordneter
                  Ordner
                </button>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Ordnerauswahl schließen"
                  onClick={() => setDirectory(null)}
                >
                  <X size={14} />
                </button>
              </div>
              {directory.children.map((child) => (
                <button
                  type="button"
                  key={child.path}
                  disabled={!!busy}
                  onClick={() => void browse(child.path)}
                >
                  <FolderOpen size={14} aria-hidden="true" /> {child.name}
                </button>
              ))}
              {!directory.children.length && (
                <p className="fine-print">Keine Unterordner.</p>
              )}
            </div>
          )}
          <p className="muted">
            lazer: Ordner mit <code>client.realm</code> und <code>files</code>.
            In Docker: <code>/osu</code>. Schließe osu! vor dem Scan.
          </p>
          <button
            type="button"
            className="primary"
            disabled={!!busy || !!scanning || !!scanReason}
            aria-busy={busy === "scan" || !!scanning}
            aria-describedby="scan-hint"
            onClick={() => void action("scan", () => api("/scan", { path }))}
          >
            <BusyIcon busy={busy === "scan" || !!scanning} icon={Search} />
            {busy === "scan"
              ? "Wird gestartet …"
              : scanning
                ? "Scan läuft …"
                : "Sammlung einlesen"}
          </button>
          <ActionHint id="scan-hint">{scanReason}</ActionHint>
          {scanning && (
            <ProgressLine
              value={scanning.progress}
              label={jobMessage(scanning) || "Sammlung wird eingelesen"}
            />
          )}
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2>
              <Upload size={18} aria-hidden="true" /> Beatmap-Dateien
            </h2>
          </div>
          <label
            className={`dropzone compact ${drop.dragging ? "dragging" : ""} ${uploading !== null ? "busy" : ""}`}
            aria-busy={uploading !== null}
            {...drop.handlers}
          >
            <input
              disabled={uploading !== null}
              type="file"
              accept={ARCHIVE_TYPES.join(",")}
              aria-label=".osz- oder .zip-Archiv auswählen"
              onChange={(event) => {
                void importArchive(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
            {uploading !== null ? (
              <LoaderCircle size={25} className="spin" aria-hidden="true" />
            ) : (
              <Upload size={25} aria-hidden="true" />
            )}
            <b>
              {uploading !== null
                ? "Archiv wird hochgeladen …"
                : drop.dragging
                  ? "Loslassen zum Importieren"
                  : ".osz oder .zip importieren"}
            </b>
            {uploading !== null ? (
              <ProgressLine value={uploading} label="Upload" />
            ) : (
              <small>Mit .osu-Dateien und zugehörigem Audio</small>
            )}
          </label>
          {importing && (
            <ProgressLine
              value={importing.progress}
              label={jobMessage(importing) || "Archiv wird importiert"}
            />
          )}
          <p className="muted">
            Es werden nur osu!standard-Maps aufgenommen. Andere Spielmodi
            bleiben außen vor.
          </p>
        </section>
        <section className="panel stars-panel">
          <div className="panel-heading">
            <h2>
              <Star size={18} aria-hidden="true" /> Star Ratings
            </h2>
            <span className="tag">offizielle osu!-Berechnung</span>
          </div>
          <p className="muted">
            stable-Songs-Ordner speichern keine Sterne. Bob berechnet sie mit
            derselben Bibliothek wie osu!lazer (Aim, Speed, Reading) und nutzt
            sie für Filter, Training und Ziel-Sterne. Neue Importe werden
            automatisch berechnet.
          </p>
          <button
            type="button"
            className="secondary"
            disabled={!state.difficulty_ready || !!busy || !!rating}
            aria-busy={busy === "stars" || !!rating}
            aria-describedby="stars-hint"
            onClick={() => void action("stars", () => api("/maps/stars", {}))}
          >
            <BusyIcon busy={busy === "stars" || !!rating} icon={Star} />
            {rating ? "Sterne werden berechnet …" : "Fehlende Sterne berechnen"}
          </button>
          <ActionHint id="stars-hint">
            {!state.difficulty_ready
              ? "Der Sterne-Rechner fehlt. Führe scripts/setup.ps1 erneut aus."
              : lastRating?.status === "completed" && lastRating.result
                ? `${lastRating.result.calculated ?? 0} Maps berechnet${lastRating.result.failed ? `, ${lastRating.result.failed} nicht berechenbar` : ""}.`
                : ""}
          </ActionHint>
          {rating && (
            <ProgressLine
              value={rating.progress}
              label={jobMessage(rating) || "Sterne werden berechnet"}
            />
          )}
        </section>
      </div>
      <section className="panel library-table" aria-busy={loading}>
        <div className="table-toolbar">
          <h2>
            Beatmaps {!error && <span className="count">{total}</span>}
          </h2>
          <div className={`search-input ${loading ? "pending" : ""}`}>
            {loading && filters.search ? (
              <LoaderCircle size={16} className="spin" aria-hidden="true" />
            ) : (
              <Search size={16} aria-hidden="true" />
            )}
            <input
              aria-label="Beatmaps suchen"
              placeholder="Song, Mapper oder Difficulty suchen …"
              value={filters.search}
              onChange={(event) => {
                updateFilters({ search: event.target.value });
              }}
              onKeyDown={(event) => {
                if (event.key === "Escape") updateFilters({ search: "" });
              }}
            />
            {filters.search && (
              <button
                type="button"
                className="icon-button search-clear"
                aria-label="Suche leeren"
                onClick={() => updateFilters({ search: "" })}
              >
                <X size={14} />
              </button>
            )}
          </div>
        </div>
        <LibraryFilters
          filters={filters}
          onChange={updateFilters}
          invalid={invalid}
        />
        <div className="selection-bar">
          <label className="checkbox-label">
            <input
              type="checkbox"
              disabled={loading || selecting || !items.length}
              checked={
                items.length > 0 && items.every((item) => selected.has(item.id))
              }
              onChange={(event) => togglePage(event.target.checked)}
            />{" "}
            Diese Seite auswählen
          </label>
          <button
            type="button"
            className="secondary"
            disabled={loading || selecting || invalid || !total || !!error}
            aria-busy={selecting}
            onClick={() => void selectAll()}
          >
            {selecting && (
              <LoaderCircle size={15} className="spin" aria-hidden="true" />
            )}
            {selecting
              ? "Auswahl wird geladen …"
              : error
                ? "Alle Treffer auswählen"
                : `Alle ${total} Treffer auswählen`}
          </button>
          <span aria-live="polite">
            {selected.size} ausgewählt, über alle Seiten
          </span>
          <button
            type="button"
            className="text-button"
            disabled={!selected.size || selecting}
            onClick={clearSelection}
          >
            Auswahl leeren
          </button>
        </div>
        <div className={`table-scroll ${loading && items.length ? "stale" : ""}`}>
          <table>
            <thead>
              <tr>
                <th aria-label="Auswahl" />
                <th>Track und Difficulty</th>
                <th>Mapper</th>
                <th>Sterne</th>
                <th>AR / CS</th>
                <th>Objekte</th>
                <th>Quelle</th>
              </tr>
            </thead>
            <tbody>
              {loading &&
                !items.length &&
                Array.from({ length: 6 }, (_, index) => (
                  <tr className="skeleton-row" key={`skeleton-${index}`} aria-hidden="true">
                    <td>
                      <i />
                    </td>
                    <td>
                      <i className="wide" />
                      <i className="short" />
                    </td>
                    <td>
                      <i />
                    </td>
                    <td>
                      <i />
                    </td>
                    <td>
                      <i />
                    </td>
                    <td>
                      <i />
                    </td>
                    <td>
                      <i />
                    </td>
                  </tr>
                ))}
              {items.map((item) => (
                <tr
                  key={item.id}
                  className={selected.has(item.id) ? "checked" : ""}
                >
                  <td>
                    <input
                      type="checkbox"
                      disabled={selecting || loading}
                      aria-label={`${item.title} ${item.version} auswählen`}
                      checked={selected.has(item.id)}
                      onChange={() => toggle(item.id)}
                    />
                  </td>
                  <td>
                    <b>{item.title}</b>
                    <small>
                      {item.artist} <span>[{item.version}]</span>
                    </small>
                  </td>
                  <td>{item.creator}</td>
                  <td
                    className="star-rating"
                    title={
                      item.star_rating == null
                        ? "Keine Sternebewertung vorhanden"
                        : "Gespeicherte Sternebewertung aus lazer"
                    }
                  >
                    {item.star_rating == null ? (
                      <span className="muted-value">keine</span>
                    ) : (
                      `${item.star_rating.toFixed(2)} ★`
                    )}
                  </td>
                  <td>
                    {item.difficulty.ApproachRate ??
                      item.difficulty.OverallDifficulty}{" "}
                    / {item.difficulty.CircleSize}
                  </td>
                  <td>{item.objects}</td>
                  <td>
                    <span className="tag">{item.source}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {error && !loading && (
          <div className="inline-error" role="alert">
            <AlertCircle size={20} aria-hidden="true" />
            <div>
              <b>Beatmaps konnten nicht geladen werden.</b>
              <p>{error}</p>
            </div>
            <button type="button" className="secondary" onClick={retry}>
              <RefreshCw size={15} aria-hidden="true" /> Erneut laden
            </button>
          </div>
        )}
        {!items.length && !loading && !invalid && !error && (
          <div className="empty-small">
            <Database size={28} aria-hidden="true" />
            <h3>
              {filtered
                ? "Keine passenden Maps"
                : "Eine leere Library voller Möglichkeiten."}
            </h3>
            <p>
              {filtered
                ? "Lockere Suche oder Sternebereich, um mehr Maps zu sehen."
                : "Lies deinen osu!-Ordner ein oder importiere ein Beatmap-Archiv."}
            </p>
            {filtered && (
              <button
                type="button"
                className="secondary"
                onClick={() =>
                  updateFilters({
                    search: "",
                    minimum: "",
                    maximum: "",
                    perSong: "all",
                  })
                }
              >
                Filter zurücksetzen
              </button>
            )}
          </div>
        )}
        <div className="pagination">
          <span aria-live="polite">
            {loading
              ? "Beatmaps werden geladen …"
              : error
                ? "Nicht geladen"
                : total
                ? `${offset + 1} bis ${lastOffset} von ${total}`
                : "0 Maps"}
          </span>
          <button
            type="button"
            className="icon-button"
            aria-label="Vorherige Seite"
            disabled={loading || offset === 0}
            onClick={() => setOffset(offset - 100)}
          >
            <ArrowLeft size={16} />
          </button>
          <button
            type="button"
            className="icon-button"
            aria-label="Nächste Seite"
            disabled={loading || offset + 100 >= total}
            onClick={() => setOffset(offset + 100)}
          >
            <ArrowRight size={16} />
          </button>
        </div>
      </section>
      <div className="panel dataset-action">
        <label>
          Name des Trainingsdatensatzes
          <input
            value={name}
            aria-invalid={!name.trim() && selected.size > 0}
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        <button
          type="button"
          className="secondary"
          disabled={!!busy || selecting || !selected.size}
          aria-busy={busy === "export"}
          aria-describedby="dataset-hint"
          onClick={() =>
            void action("export", () => api("/export", { ids: [...selected] }))
          }
        >
          <BusyIcon busy={busy === "export"} icon={Download} size={16} />
          {busy === "export" ? "Export startet …" : "Auswahl als .osz"}
        </button>
        <button
          type="button"
          className="primary"
          disabled={!!busy || selecting || !!datasetReason}
          aria-busy={busy === "dataset"}
          aria-describedby="dataset-hint"
          onClick={() =>
            void action("dataset", () =>
              api("/datasets", { ids: [...selected], name }),
            )
          }
        >
          <BusyIcon busy={busy === "dataset"} icon={Database} size={16} />
          {busy === "dataset"
            ? "Wird übernommen …"
            : selected.size
              ? `${selected.size} ${selected.size === 1 ? "Map" : "Maps"} als Trainingsdaten übernehmen`
              : "Als Trainingsdaten übernehmen"}
        </button>
        <ActionHint id="dataset-hint">{datasetReason}</ActionHint>
      </div>
      <p className="fine-print">
        Export enthält Beatmaps und Audio. Bilder, Storyboards und
        benutzerdefinierte Hitsounds werden nicht kopiert.
      </p>
    </>
  );
}
