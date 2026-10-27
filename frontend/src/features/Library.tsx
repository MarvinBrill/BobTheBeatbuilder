import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Database,
  Download,
  FolderOpen,
  Search,
  Upload,
} from "lucide-react";
import { api, upload } from "../api";
import type { State } from "../types";

import { useLibraryMaps } from "./useLibraryMaps";
import { LibraryFilters } from "./LibraryFilters";

interface Directory {
  path: string;
  parent: string;
  children: { name: string; path: string }[];
}

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
    .filter((job) => ["scan", "import"].includes(job.kind))
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
  } = useLibraryMaps(revision, onError);
  const [name, setName] = useState("Meine Mapping-Library");
  const [busy, setBusy] = useState(false);

  async function action(operation: () => Promise<unknown>) {
    setBusy(true);
    try {
      await operation();
      refresh();
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function browse(target: string) {
    try {
      const result = await api<Directory>(
        `/browse?path=${encodeURIComponent(target)}`,
      );
      setDirectory(result);
      setPath(result.path);
    } catch (error) {
      onError((error as Error).message);
    }
  }

  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">GUTE MAPS. GUTE GRUNDLAGE.</div>
          <h1>
            Deine <em>Library.</em>
          </h1>
          <p>
            Importiere deine osu!-Sammlung und kuratiere deinen Trainingsstil.
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
              <FolderOpen size={18} /> Aus osu! importieren
            </h2>
            <span className="tag">lazer & stable</span>
          </div>
          <label>
            osu!-Datenordner oder Songs-Verzeichnis
            <div className="input-action">
              <input
                value={path}
                onChange={(event) => setPath(event.target.value)}
              />
              <button className="secondary" onClick={() => void browse(path)}>
                Durchsuchen
              </button>
            </div>
          </label>
          {directory && (
            <div className="directory-browser">
              <button onClick={() => void browse(directory.parent)}>
                <ArrowLeft size={14} /> Übergeordneter Ordner
              </button>
              {directory.children.map((child) => (
                <button
                  key={child.path}
                  onClick={() => void browse(child.path)}
                >
                  <FolderOpen size={14} /> {child.name}
                </button>
              ))}
            </div>
          )}
          <p className="muted">
            lazer: Ordner mit <code>client.realm</code> und <code>files</code>.
            In Docker: <code>/osu</code>. Schließe osu! vor dem Scan.
          </p>
          <button
            className="primary"
            disabled={busy || !path}
            onClick={() => void action(() => api("/scan", { path }))}
          >
            <Search size={17} /> Sammlung einlesen
          </button>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2>
              <Upload size={18} /> Beatmap-Dateien
            </h2>
          </div>
          <label className="dropzone compact">
            <input
              disabled={busy}
              type="file"
              accept=".osz,.zip"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void action(() => upload("/import", file));
                event.target.value = "";
              }}
            />
            <Upload size={25} />
            <b>.osz oder .zip importieren</b>
            <small>Mit .osu-Dateien und zugehörigem Audio</small>
          </label>
          <p className="muted">
            Es werden nur osu!standard-Maps aufgenommen. Andere Spielmodi
            bleiben außen vor.
          </p>
        </section>
      </div>
      <section className="panel library-table">
        <div className="table-toolbar">
          <h2>
            Beatmaps <span className="count">{total}</span>
          </h2>
          <div className="search-input">
            <Search size={16} />
            <input
              aria-label="Beatmaps suchen"
              placeholder="Song, Mapper oder Difficulty suchen …"
              value={filters.search}
              onChange={(event) => {
                updateFilters({ search: event.target.value });
              }}
            />
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
              disabled={loading || selecting}
              checked={
                items.length > 0 && items.every((item) => selected.has(item.id))
              }
              onChange={(event) => togglePage(event.target.checked)}
            />{" "}
            Diese Seite auswählen
          </label>
          <button
            className="secondary"
            disabled={loading || selecting || invalid || !total}
            onClick={() => void selectAll()}
          >
            {selecting
              ? "Auswahl wird geladen …"
              : `Alle ${total} Treffer auswählen`}
          </button>
          <span>{selected.size} ausgewählt · über alle Seiten</span>
          <button className="text-button" onClick={clearSelection}>
            Auswahl leeren
          </button>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th aria-label="Auswahl" />
                <th>TRACK / DIFFICULTY</th>
                <th>MAPPER</th>
                <th>STERNE</th>
                <th>AR / CS</th>
                <th>OBJEKTE</th>
                <th>QUELLE</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr
                  key={item.id}
                  className={selected.has(item.id) ? "checked" : ""}
                >
                  <td>
                    <input
                      type="checkbox"
                      disabled={selecting}
                      aria-label={`${item.title} ${item.version} auswählen`}
                      checked={selected.has(item.id)}
                      onChange={() => toggle(item.id)}
                    />
                  </td>
                  <td>
                    <b>{item.title}</b>
                    <small>
                      {item.artist} <span>· {item.version}</span>
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
                    {item.star_rating == null
                      ? "—"
                      : `${item.star_rating.toFixed(2)} ★`}
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
        {!items.length && !loading && !invalid && (
          <div className="empty-small">
            <Database size={28} />
            <h3>
              {filters.search ||
              filters.minimum ||
              filters.maximum ||
              filters.perSong !== "all"
                ? "Keine passenden Maps"
                : "Eine leere Library voller Möglichkeiten."}
            </h3>
            <p>
              Lies deinen osu!-Ordner ein oder importiere ein Beatmap-Archiv.
            </p>
          </div>
        )}
        {loading && (
          <p className="fine-print" role="status">
            Beatmaps werden geladen …
          </p>
        )}
        <div className="pagination">
          <span>
            {total
              ? `${offset + 1}–${Math.min(offset + 100, total)} von ${total}`
              : "0 Maps"}
          </span>
          <button
            className="icon-button"
            aria-label="Vorherige Seite"
            disabled={loading || offset === 0}
            onClick={() => setOffset(offset - 100)}
          >
            <ArrowLeft size={16} />
          </button>
          <button
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
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        <button
          className="secondary"
          disabled={busy || selecting || !selected.size}
          onClick={() =>
            void action(() => api("/export", { ids: [...selected] }))
          }
        >
          <Download size={16} /> Auswahl als .osz
        </button>
        <button
          className="primary"
          disabled={busy || selecting || !selected.size || !name.trim()}
          onClick={() =>
            void action(() => api("/datasets", { ids: [...selected], name }))
          }
        >
          <Database size={16} /> Als Trainingsdaten übernehmen
        </button>
      </div>
      <p className="fine-print">
        Export enthält Beatmaps und Audio. Bilder, Storyboards und
        benutzerdefinierte Hitsounds werden nicht kopiert.
      </p>
    </>
  );
}
