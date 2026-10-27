import { useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { LoaderCircle, Trash2 } from "lucide-react";
import { deleteArtifact } from "../api";

export function DeleteArtifactButton({
  kind,
  identifier,
  name,
  blocked,
  onDeleted,
}: {
  kind: "model" | "output";
  identifier: string;
  name: string;
  blocked: boolean;
  onDeleted: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const heading = useId();
  const description = useId();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const label = kind === "model" ? "Modell" : "Beatmap";

  async function remove() {
    setBusy(true);
    setError("");
    try {
      await deleteArtifact(kind, identifier);
      dialog.current?.close();
      onDeleted();
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <button
        className="icon-button delete-button"
        aria-label={`${label} „${name}“ löschen`}
        title={
          blocked
            ? "Zuerst laufende oder wartende Aufgaben beenden"
            : `${label} löschen`
        }
        disabled={blocked || busy}
        onClick={() => {
          setError("");
          dialog.current?.showModal();
        }}
      >
        <Trash2 size={17} />
      </button>
      {createPortal(
        <dialog
          ref={dialog}
          className="delete-dialog"
          aria-labelledby={heading}
          aria-describedby={description}
          onCancel={(event) => {
            if (busy) event.preventDefault();
          }}
        >
          <h2 id={heading}>{label} endgültig löschen?</h2>
          <p className="delete-name">{name}</p>
          <div id={description}>
            <p>
              {kind === "model"
                ? "Gewichte, Trainingsstände, Modell-ZIP und zugehörige Verlaufseinträge werden von der Festplatte entfernt."
                : "Die Beatmap wird mit Export-ZIP, Vorschau, ihrer Audiokopie und zugehörigen Verlaufseinträgen von der Festplatte entfernt."}
            </p>
            <p>
              {kind === "model"
                ? "Trainingsdatensätze und bereits erzeugte Beatmaps bleiben erhalten."
                : "Das verwendete Modell, Trainingsdaten und ursprüngliche Audio-Uploads bleiben erhalten."}
            </p>
            <p>Das Löschen kann nicht rückgängig gemacht werden.</p>
            <p>
              Analyse-Caches werden ebenfalls geleert. Weitere Trainingskopien
              und Uploads verwaltest du unter „Speicherverwaltung“.
            </p>
          </div>
          {blocked && (
            <p className="delete-error" role="status">
              Zuerst laufende oder wartende Aufgaben beenden.
            </p>
          )}
          {error && (
            <p className="delete-error" role="alert">
              {error}
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
              {busy ? (
                <LoaderCircle size={17} className="spin" />
              ) : (
                <Trash2 size={17} />
              )}
              {busy ? "Wird gelöscht …" : "Endgültig löschen"}
            </button>
          </div>
        </dialog>,
        document.body,
      )}
    </>
  );
}
