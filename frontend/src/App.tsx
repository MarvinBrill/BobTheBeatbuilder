import { useCallback, useEffect, useState } from "react";
import {
  AudioLines,
  BrainCircuit,
  CircleHelp,
  Database,
  HardDrive,
  Layers,
  Sparkles,
  X,
} from "lucide-react";
import { api } from "./api";
import { JobActivity } from "./components/JobActivity";
import { Studio } from "./features/Studio";
import { Library } from "./features/Library";
import { Training } from "./features/Training";
import { Models } from "./features/Models";
import { Storage } from "./features/Storage";
import { useStatePolling } from "./features/useStatePolling";
import type { Hardware, Page, State } from "./types";

export default function App() {
  const [page, setPage] = useState<Page>("studio");
  const [state, setState] = useState<State | null>(null);
  const [hardware, setHardware] = useState<Hardware | null>(null);
  const [error, setError] = useState("");
  const [help, setHelp] = useState(false);
  const onError = useCallback((message: string) => setError(message), []);
  const refresh = useStatePolling(setState, onError);
  const onDeleted = useCallback(
    (kind: "model" | "output", identifier: string) => {
      setState((current) =>
        current
          ? {
              ...current,
              models:
                kind === "model"
                  ? current.models.filter((item) => item.id !== identifier)
                  : current.models,
              outputs:
                kind === "output"
                  ? current.outputs.filter((item) => item.id !== identifier)
                  : current.outputs,
            }
          : current,
      );
      refresh();
    },
    [refresh],
  );

  useEffect(() => {
    void api<Hardware>("/hardware")
      .then(setHardware)
      .catch((error) => onError(error.message));
  }, [onError]);
  const navigation = [
    { id: "studio", title: "Beatmap Studio", icon: AudioLines },
    { id: "library", title: "Beatmap Library", icon: Database },
    { id: "training", title: "Model Training", icon: BrainCircuit },
    { id: "models", title: "Meine Modelle", icon: Layers },
    { id: "storage", title: "Speicherverwaltung", icon: HardDrive },
  ] as const;
  const props = state ? { state, hardware, refresh, onError, onDeleted } : null;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(event) => {
            event.preventDefault();
            setPage("studio");
          }}
        >
          <div className="brand-mark">
            b<span>•</span>
          </div>
          <div>
            bob<span>THE BEATBUILDER</span>
          </div>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <nav>
          {navigation.map((item) => (
            <button
              key={item.id}
              className={page === item.id ? "active" : ""}
              onClick={() => setPage(item.id)}
            >
              <item.icon size={19} />
              {item.title}
              {page === item.id && <i />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-card">
            <HardDrive size={18} />
            <b>Made to stay local.</b>
            <p>
              Deine Daten.
              <br />
              Deine GPU. Dein Flow.
            </p>
            <span>
              <span className="status-dot" />{" "}
              {state ? "Backend verbunden" : "Verbindung wird aufgebaut"}
            </span>
          </div>
          <button className="help-button" onClick={() => setHelp(true)}>
            <CircleHelp size={17} /> So funktioniert’s <span>↗</span>
          </button>
          <div className="sidebar-version">
            BOB STUDIO <span>v0.1 / EXPERIMENTAL</span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div>
            Workspace <span>/</span>
            <b>{navigation.find((item) => item.id === page)?.title}</b>
          </div>
          <div className="topbar-status">
            <span className="status-dot" /> {hardware?.name || "Local engine"}
          </div>
        </header>
        <main>
          {props ? (
            <>
              {page === "studio" && (
                <Studio {...props} goLibrary={() => setPage("library")} />
              )}
              {page === "library" && <Library {...props} />}
              {page === "storage" && <Storage {...props} />}
              {page === "training" && <Training {...props} />}
              {page === "models" && (
                <Models
                  onDeleted={onDeleted}
                  refresh={refresh}
                  onError={onError}
                  state={props.state}
                  goTraining={() => setPage("training")}
                />
              )}
              <section className="panel activity-panel">
                <div className="panel-heading">
                  <h2>
                    <Sparkles size={16} /> Aktivität
                  </h2>
                  <span className="tag">
                    {
                      state?.jobs.filter((job) =>
                        ["queued", "running"].includes(job.status),
                      ).length
                    }{" "}
                    aktiv
                  </span>
                </div>
                <JobActivity jobs={props.state.jobs} onError={onError} />
              </section>
            </>
          ) : (
            <div className="large-empty">
              <AudioLines className="spin" size={40} />
              <h2>Studio wird verbunden …</h2>
              <p>Das lokale Backend muss auf Port 8000 laufen.</p>
              <button className="secondary" onClick={refresh}>
                Erneut verbinden
              </button>
            </div>
          )}
          <footer>
            <span>Built around your rhythm.</span>
            <span>osu!standard · Local AI · No cloud required</span>
          </footer>
        </main>
      </div>
      {error && (
        <div className="toast" role="alert">
          <div>
            <b>Das hat nicht geklappt</b>
            <p>{error}</p>
          </div>
          <button
            className="icon-button"
            aria-label="Fehlermeldung schließen"
            onClick={() => setError("")}
          >
            <X size={18} />
          </button>
        </div>
      )}
      {help && (
        <div className="modal-backdrop" onClick={() => setHelp(false)}>
          <section
            className="panel help-modal"
            role="dialog"
            aria-modal="true"
            aria-label="So funktioniert Bob"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="panel-heading">
              <h2>Dein Weg zur ersten Beatmap</h2>
              <button
                className="icon-button"
                aria-label="Schließen"
                onClick={() => setHelp(false)}
              >
                <X size={18} />
              </button>
            </div>
            <ol>
              <li>
                <b>Library:</b> osu!-Datenordner einlesen oder .osz importieren.
              </li>
              <li>
                <b>Auswählen:</b> Passende Maps als Trainingsdatensatz
                übernehmen.
              </li>
              <li>
                <b>Training:</b> Modellname und Rechengerät wählen, Training
                starten.
              </li>
              <li>
                <b>Studio:</b> Audio und Modell wählen, Parameter einstellen und
                generieren.
              </li>
              <li>
                <b>Spielen:</b> .osz herunterladen und mit osu!lazer öffnen.
              </li>
            </ol>
            <p className="muted">
              Für brauchbare Ergebnisse viele unterschiedliche Songs mit
              konsistentem Mapping-Stil verwenden. Ein kurzer Testlauf prüft den
              Ablauf, nicht die Mapping-Qualität. Automatisches Timing verwendet
              eine konstante BPM-Schätzung; komplexe Tempowechsel brauchen
              Nachbearbeitung.
            </p>
          </section>
        </div>
      )}
    </div>
  );
}
