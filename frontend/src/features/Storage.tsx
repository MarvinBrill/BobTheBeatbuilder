import { useCallback, useEffect, useRef, useState } from "react";
import { HardDrive, RefreshCw, Trash2 } from "lucide-react";
import { api } from "../api";
import type { State } from "../types";

interface Entry {
  kind: string;
  id: string;
  name: string;
  bytes: number;
  detail: string;
  removable: boolean;
  paths: string[];
}
interface Inventory {
  entries: Entry[];
  total_bytes: number;
  cache_limit_gib: number;
  blocked: boolean;
}
const groups: Record<string, string> = {
  model: "Modelle · einschließlich älterer Versionen",
  dataset: "Trainingsdatensätze",
  output: "Generierte Beatmaps und Exporte",
  audio: "Audio-Uploads",
  import: "Lokal importierte Bibliothek",
  catalog: "Bibliothekskatalog",
  history: "Aufgabenverlauf",
  cleanup: "Caches und temporäre Dateien",
  orphan: "Nicht zugeordnete Dateien",
  development: "Test- und Benchmark-Dateien",
  runtime: "Laufzeit und sonstige Dateien · nur Anzeige",
};
function size(bytes: number) {
  return bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(2)} GiB`
    : `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

export function Storage({
  state,
  refresh,
  onError,
}: {
  state: State;
  refresh: () => void;
  onError: (message: string) => void;
}) {
  const [report, setReport] = useState<Inventory | null>(null);
  const [limit, setLimit] = useState("1");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<Entry | null>(null);
  const [failure, setFailure] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const blocked = state.deletion_blocked || !!report?.blocked;
  const load = useCallback(async () => {
    setLoading(true);
    try {
      const next = await api<Inventory>("/storage");
      setReport(next);
      setLimit(String(next.cache_limit_gib));
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setLoading(false);
    }
  }, [onError]);
  useEffect(() => {
    void load();
  }, [load, state.deletion_blocked]);

  async function saveLimit() {
    setBusy(true);
    try {
      await api("/storage/settings", { cache_limit_gib: Number(limit) });
      await load();
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function remove() {
    if (!pending) return;
    setBusy(true);
    setFailure("");
    try {
      const response = await fetch(
        `/api/storage/${encodeURIComponent(pending.kind)}/${encodeURIComponent(pending.id)}`,
        { method: "DELETE" },
      );
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Löschen fehlgeschlagen");
      }
      dialog.current?.close();
      setPending(null);
      refresh();
      await load();
    } catch (error) {
      setFailure((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">DEINE DATEIEN. DEIN PLATZ.</div>
          <h1>
            Dein <em>Speicher.</em>
          </h1>
          <p>
            Alle Dateien im data-Ordner – mit Größe und gezielter Bereinigung.
          </p>
        </div>
        <button
          className="secondary"
          disabled={loading || busy}
          onClick={() => void load()}
        >
          <RefreshCw size={17} /> Aktualisieren
        </button>
      </div>
      <section className="panel storage-summary">
        <div>
          <HardDrive size={25} />
          <strong>
            {report ? size(report.total_bytes) : "Wird gemessen …"}
          </strong>
          <span>im data-Ordner</span>
        </div>
        <p>
          Modelle, Trainingskopien und ursprüngliche Uploads sind eigenständige
          Einträge. Lösche sie hier einzeln, wenn du sie nicht mehr benötigst.
          Externe osu!-Originaldateien werden niemals gelöscht.
        </p>
        <div className="storage-cache-setting">
          <label>
            Analyse-Cache maximal (GiB)
            <input
              type="number"
              min="0"
              max="64"
              step="0.25"
              value={limit}
              onChange={(event) => setLimit(event.target.value)}
            />
          </label>
          <button
            className="secondary"
            disabled={
              busy ||
              blocked ||
              loading ||
              limit === "" ||
              !Number.isFinite(Number(limit)) ||
              Number(limit) < 0 ||
              Number(limit) > 64
            }
            onClick={() => void saveLimit()}
          >
            Limit speichern
          </button>
        </div>
        <p className="fine-print">
          Automatisch nach jeder Aufgabe begrenzt; während des Trainings darf
          der Cache größer sein. 0 entfernt alle Analyse-Caches nach der
          Aufgabe. Caches ändern keine Modellgewichte oder Ergebnisse, nur die
          Vorbereitungszeit. Download-Caches sind separat über „Caches leeren“
          entfernbar.
        </p>
        {blocked && (
          <p role="status">
            Löschen und Bereinigen ist während Aufgaben oder Uploads gesperrt.
            Nach deren Ende wird die Ansicht aktualisiert.
          </p>
        )}
      </section>
      {Object.entries(groups).map(([kind, title]) => {
        const rows =
          report?.entries.filter((entry) => entry.kind === kind) || [];
        if (!rows.length) return null;
        return (
          <section className="panel storage-group" key={kind}>
            <div className="panel-heading">
              <h2>{title}</h2>
              <span className="tag">
                {size(rows.reduce((sum, entry) => sum + entry.bytes, 0))}
              </span>
            </div>
            {rows.map((entry) => (
              <div className="storage-row" key={entry.id}>
                <div>
                  <b>{entry.name}</b>
                  <p>{entry.detail}</p>
                  <small>
                    {entry.paths.slice(0, 2).join(" · ") ||
                      "Nur Katalogeinträge"}
                    {entry.paths.length > 2
                      ? ` · und ${entry.paths.length - 2} weitere`
                      : ""}
                  </small>
                </div>
                <span className="storage-size">{size(entry.bytes)}</span>
                {entry.removable && (
                  <button
                    className="icon-button delete-button"
                    aria-label={`${entry.name} löschen`}
                    disabled={busy || blocked || loading}
                    onClick={() => {
                      setPending(entry);
                      setFailure("");
                      dialog.current?.showModal();
                    }}
                  >
                    <Trash2 size={18} />
                  </button>
                )}
              </div>
            ))}
          </section>
        );
      })}
      <dialog
        ref={dialog}
        className="delete-dialog"
        aria-labelledby="storage-delete-title"
        aria-describedby="storage-delete-detail"
        onCancel={(event) => {
          if (busy) event.preventDefault();
        }}
      >
        <h2 id="storage-delete-title">Endgültig vom Datenträger entfernen?</h2>
        <p className="delete-name">
          {pending?.name} · {size(pending?.bytes || 0)}
        </p>
        <div id="storage-delete-detail">
          <p>{pending?.detail}</p>
          {!!pending?.paths.length && (
            <details className="storage-paths">
              <summary>{pending.paths.length} Speicherpfade anzeigen</summary>
              <ul>
                {pending.paths.map((path) => (
                  <li key={path}>{path}</li>
                ))}
              </ul>
            </details>
          )}
          <p>
            Das Löschen kann nicht rückgängig gemacht werden. Gemeinsam
            berechnete Analyse-Caches werden bei Artefaktlöschungen ebenfalls
            geleert und bei Bedarf neu erstellt.
          </p>
        </div>
        {failure && (
          <p className="delete-error" role="alert">
            {failure}
          </p>
        )}
        <div className="dialog-actions">
          <button
            autoFocus
            className="secondary"
            disabled={busy}
            onClick={() => dialog.current?.close()}
          >
            Abbrechen
          </button>
          <button
            className="danger-button"
            disabled={busy || blocked}
            onClick={() => void remove()}
          >
            {busy ? "Wird entfernt …" : "Endgültig löschen"}
          </button>
        </div>
      </dialog>
    </>
  );
}
