import { useState } from "react";
import { ArrowLeft, ArrowRight, AudioLines, Download } from "lucide-react";
import { DeleteArtifactButton } from "../components/DeleteArtifactButton";
import type { Output } from "../types";

const PAGE_SIZE = 10;

export function GeneratedMaps({
  outputs,
  selectedId,
  blocked,
  onSelect,
  onDeleted,
}: {
  outputs: Output[];
  selectedId?: string;
  blocked: boolean;
  onSelect: (identifier: string) => void;
  onDeleted: (identifier: string) => void;
}) {
  const [page, setPage] = useState(0);
  const lastPage = Math.max(0, Math.ceil(outputs.length / PAGE_SIZE) - 1);
  const currentPage = Math.min(page, lastPage);
  const offset = currentPage * PAGE_SIZE;
  return (
    <section className="panel recent-panel">
      <div className="panel-heading">
        <h2>Deine Kreationen</h2>
        <span className="tag">
          {outputs.length} {outputs.length === 1 ? "Map" : "Maps"}
        </span>
      </div>
      {blocked && (
        <p className="fine-print">
          Löschen ist nach Abschluss oder Abbruch aller Aufgaben möglich.
        </p>
      )}
      {outputs.length ? (
        <>
          {outputs.slice(offset, offset + PAGE_SIZE).map((output) => (
            <div className="output-row" key={output.id}>
              <button
                type="button"
                className="text-button"
                aria-pressed={output.id === selectedId}
                onClick={() => onSelect(output.id)}
              >
                <AudioLines size={18} aria-hidden="true" />
                <span>
                  {output.name}
                  <small>
                    {output.objects} Objekte, {output.bpm} BPM
                  </small>
                </span>
              </button>
              <div className="artifact-actions">
                <a
                  className="icon-button"
                  aria-label={`${output.name} herunterladen`}
                  href={`/api/outputs/${output.id}/download`}
                >
                  <Download size={17} />
                </a>
                <DeleteArtifactButton
                  kind="output"
                  identifier={output.id}
                  name={output.name}
                  blocked={blocked}
                  onDeleted={() => onDeleted(output.id)}
                />
              </div>
            </div>
          ))}
          <div className="pagination">
            <span>
              {offset + 1} bis {Math.min(offset + PAGE_SIZE, outputs.length)} von{" "}
              {outputs.length}
            </span>
            <button
              type="button"
              className="icon-button"
              aria-label="Vorherige Kreationen"
              disabled={!currentPage}
              onClick={() => setPage(currentPage - 1)}
            >
              <ArrowLeft size={16} />
            </button>
            <button
              type="button"
              className="icon-button"
              aria-label="Weitere Kreationen"
              disabled={currentPage >= lastPage}
              onClick={() => setPage(currentPage + 1)}
            >
              <ArrowRight size={16} />
            </button>
          </div>
        </>
      ) : (
        <div className="empty-small">
          Noch unbespielt.
          <p>Deine generierten Maps warten hier auf ihren ersten Full Combo.</p>
        </div>
      )}
    </section>
  );
}
