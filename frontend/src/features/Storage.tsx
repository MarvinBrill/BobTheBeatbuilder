import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Check,
  HardDrive,
  LoaderCircle,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { api, remove as removeRequest } from "../api";
import { ActionHint, BusyIcon } from "../components/Feedback";
import { useToast } from "../components/Toasts";
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
  const { notify } = useToast();
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [busy, setBusy] = useState<"" | "limit" | "delete">("");
  const [saved, setSaved] = useState(false);
  const [pending, setPending] = useState<Entry | null>(null);
  const [failure, setFailure] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const blocked = state.deletion_blocked || !!report?.blocked;
  const load = useCallback(async () => {
    setLoading(true);
    setLoadError("");
    try {
      const next = await api<Inventory>("/storage");
      setReport(next);
      setLimit(String(next.cache_limit_gib));
    } catch (error) {
      setLoadError((error as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);
  const limitValue = Number(limit);
  const limitInvalid =
    limit === "" ||
    !Number.isFinite(limitValue) ||
    limitValue < 0 ||
    limitValue > 64;
  const limitUnchanged =
    !!report && !limitInvalid && limitValue === report.cache_limit_gib;
  useEffect(() => {
    if (!saved) return;
    const timer = setTimeout(() => setSaved(false), 2500);
    return () => clearTimeout(timer);
  }, [saved]);
  useEffect(() => {
    void load();
  }, [load, state.deletion_blocked]);

  async function saveLimit() {
    setBusy("limit");
    setSaved(false);
    try {
      await api("/storage/settings", { cache_limit_gib: limitValue });
      await load();
      setSaved(true);
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setBusy("");
    }
  }
  async function remove() {
    if (!pending) return;
    const entry = pending;
    setBusy("delete");
    setFailure("");
    try {
      await removeRequest(
        `/storage/${encodeURIComponent(entry.kind)}/${encodeURIComponent(entry.id)}`,
      );
      dialog.current?.close();
      setPending(null);
      notify({
        tone: "success",
        title: "Vom Datenträger entfernt",
        message: `„${entry.name}“, ${size(entry.bytes)} freigegeben.`,
      });
      refresh();
      await load();
    } catch (error) {
      setFailure((error as Error).message);
    } finally {
      setBusy("");
    }
  }

  return (
    <>
      <div className="page-heading">
        <div>
          <h1>
            Dein <span className="accent">Speicher.</span>
          </h1>
          <p>
            Deine Dateien, dein Platz. Alle Dateien im data-Ordner, mit Größe
            und gezielter Bereinigung.
          </p>
        </div>
        <button
          type="button"
          className="secondary"
          disabled={loading || !!busy}
          aria-busy={loading}
          onClick={() => void load()}
        >
          <RefreshCw
            size={17}
            className={loading ? "spin" : ""}
            aria-hidden="true"
          />
          {loading ? "Wird gemessen …" : "Aktualisieren"}
        </button>
      </div>
      <section className="panel storage-summary">
        <div>
          <HardDrive size={25} aria-hidden="true" />
          <strong>
            {report
              ? size(report.total_bytes)
              : loadError
                ? "Unbekannt"
                : "Wird gemessen …"}
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
              aria-invalid={limitInvalid}
              aria-describedby="limit-hint"
              onChange={(event) => {
                setLimit(event.target.value);
                setSaved(false);
              }}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !busy &&
                  !blocked &&
                  !loading &&
                  !limitInvalid &&
                  !limitUnchanged
                )
                  void saveLimit();
              }}
            />
          </label>
          <button
            type="button"
            className={`secondary ${saved ? "is-saved" : ""}`}
            disabled={
              !!busy || blocked || loading || limitInvalid || limitUnchanged
            }
            aria-busy={busy === "limit"}
            onClick={() => void saveLimit()}
          >
            {busy === "limit" ? (
              <LoaderCircle size={16} className="spin" aria-hidden="true" />
            ) : saved ? (
              <Check size={16} aria-hidden="true" />
            ) : null}
            {busy === "limit"
              ? "Wird gespeichert …"
              : saved
                ? "Gespeichert"
                : "Limit speichern"}
          </button>
        </div>
        <ActionHint
          id="limit-hint"
          tone={limitInvalid ? "error" : "neutral"}
        >
          {limitInvalid ? "Gib einen Wert von 0 bis 64 GiB ein." : ""}
        </ActionHint>
        <p className="fine-print">
          Automatisch nach jeder Aufgabe begrenzt; während des Trainings darf
          der Cache größer sein. 0 entfernt alle Analyse-Caches nach der
          Aufgabe. Caches ändern keine Modellgewichte oder Ergebnisse, nur die
          Vorbereitungszeit. Download-Caches sind separat über „Caches leeren“
          entfernbar.
        </p>
        {blocked && (
          <p role="status" className="blocked-note">
            Löschen und Bereinigen ist während Aufgaben oder Uploads gesperrt.
            Nach deren Ende wird die Ansicht aktualisiert.
          </p>
        )}
      </section>
      {loadError && (
        <div className="panel inline-error" role="alert">
          <AlertCircle size={20} aria-hidden="true" />
          <div>
            <b>Der Speicher konnte nicht gemessen werden.</b>
            <p>{loadError}</p>
          </div>
          <button
            type="button"
            className="secondary"
            disabled={loading}
            onClick={() => void load()}
          >
            <RefreshCw size={15} aria-hidden="true" /> Erneut messen
          </button>
        </div>
      )}
      {!report && loading && (
        <section className="panel storage-group" aria-hidden="true">
          {Array.from({ length: 3 }, (_, index) => (
            <div className="storage-row skeleton-block" key={index}>
              <div>
                <i className="wide" />
                <i className="short" />
              </div>
            </div>
          ))}
        </section>
      )}
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
                    {entry.paths.slice(0, 2).join(", ") ||
                      "Nur Katalogeinträge"}
                    {entry.paths.length > 2
                      ? ` und ${entry.paths.length - 2} weitere`
                      : ""}
                  </small>
                </div>
                <span className="storage-size">{size(entry.bytes)}</span>
                {entry.removable && (
                  <button
                    className="icon-button delete-button"
                    aria-label={`${entry.name} löschen`}
                    disabled={!!busy || blocked || loading}
                    title={
                      blocked
                        ? "Zuerst laufende oder wartende Aufgaben beenden"
                        : undefined
                    }
                    onClick={() => {
                      setPending(entry);
                      setFailure("");
                      dialog.current?.showModal();
                    }}
                  >
                    <Trash2 size={18} aria-hidden="true" />
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
          {pending?.name} ({size(pending?.bytes || 0)})
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
            type="button"
            autoFocus
            className="secondary"
            disabled={!!busy}
            onClick={() => dialog.current?.close()}
          >
            Abbrechen
          </button>
          <button
            type="button"
            className="danger-button"
            disabled={!!busy || blocked}
            aria-busy={busy === "delete"}
            onClick={() => void remove()}
          >
            <BusyIcon busy={busy === "delete"} icon={Trash2} />
            {busy === "delete" ? "Wird entfernt …" : "Endgültig löschen"}
          </button>
        </div>
      </dialog>
    </>
  );
}
