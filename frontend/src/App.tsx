import { useCallback, useEffect, useRef, useState } from "react";
import {
  AudioLines,
  BrainCircuit,
  CircleHelp,
  Database,
  HardDrive,
  Layers,
  LoaderCircle,
  RefreshCw,
  Sparkles,
  X,
} from "lucide-react";
import { api } from "./api";
import { JobActivity } from "./components/JobActivity";
import { useToast } from "./components/Toasts";
import { Studio } from "./features/Studio";
import { Library } from "./features/Library";
import { Training } from "./features/Training";
import { Models } from "./features/Models";
import { Storage } from "./features/Storage";
import { useStatePolling } from "./features/useStatePolling";
import { useJobNotifications } from "./features/useJobNotifications";
import { isActive, jobTitle, scrollToActivity, statusLabel } from "./features/jobs";
import type { Hardware, Page, State } from "./types";

const navigation = [
  { id: "studio", title: "Beatmap Studio", icon: AudioLines },
  { id: "library", title: "Beatmap Library", icon: Database },
  { id: "training", title: "Model Training", icon: BrainCircuit },
  { id: "models", title: "Meine Modelle", icon: Layers },
  { id: "storage", title: "Speicherverwaltung", icon: HardDrive },
] as const;

/** The page lives in the URL (#/library), so reload and Back keep it. */
function pageFromHash(): Page {
  const id = window.location.hash.replace(/^#\/?/, "");
  return navigation.some((item) => item.id === id) ? (id as Page) : "studio";
}

type Connection = "connecting" | "online" | "offline";
const connectionText: Record<Connection, string> = {
  connecting: "Verbindung wird aufgebaut …",
  online: "Backend verbunden",
  offline: "Backend nicht erreichbar",
};

export default function App() {
  const { notify, dismiss } = useToast();
  const [page, setPage] = useState<Page>(pageFromHash);
  const [state, setState] = useState<State | null>(null);
  const [hardware, setHardware] = useState<Hardware | null>(null);
  const [connection, setConnection] = useState<Connection>("connecting");
  const [retrying, setRetrying] = useState(false);
  const [shownOutput, setShownOutput] = useState("");
  const connectionRef = useRef<Connection>("connecting");
  const refreshRef = useRef<() => void>(() => {});
  const help = useRef<HTMLDialogElement>(null);

  const navigate = useCallback((next: Page) => {
    if (window.location.hash !== `#/${next}`) window.location.hash = `/${next}`;
    setPage(next);
    window.scrollTo({ top: 0 });
  }, []);
  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const onError = useCallback(
    (message: string) =>
      notify({ tone: "error", title: "Das hat nicht geklappt", message }),
    [notify],
  );
  const onState = useCallback(
    (next: State) => {
      setState(next);
      setRetrying(false);
      if (connectionRef.current === "offline") {
        dismiss("connection");
        notify({
          key: "connection-back",
          tone: "success",
          title: "Verbindung wiederhergestellt",
        });
      }
      connectionRef.current = "online";
      setConnection("online");
    },
    [notify, dismiss],
  );
  const onPollError = useCallback(() => {
    setRetrying(false);
    connectionRef.current = "offline";
    setConnection("offline");
    notify({
      key: "connection",
      tone: "error",
      title: "Backend nicht erreichbar",
      message:
        "Bob versucht es automatisch weiter. Läuft das lokale Backend auf Port 8000?",
      action: { label: "Erneut versuchen", run: () => refreshRef.current() },
    });
  }, [notify]);
  const refresh = useStatePolling(onState, onPollError);
  refreshRef.current = refresh;

  const showOutput = useCallback((identifier: string) => {
    setShownOutput(identifier);
  }, []);
  useJobNotifications(state?.jobs, notify, navigate, showOutput, page);

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

  const props = state ? { state, hardware, refresh, onError, onDeleted } : null;
  const running = state?.jobs.filter(isActive) || [];
  const progress =
    running.reduce((sum, job) => sum + (job.progress || 0), 0) /
    Math.max(1, running.length);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#/studio"
          onClick={(event) => {
            event.preventDefault();
            navigate("studio");
          }}
        >
          <div className="brand-mark">
            b<span>•</span>
          </div>
          <div>
            bob<span>the beatbuilder</span>
          </div>
        </a>
        <div className="workspace-label">Workspace</div>
        <nav aria-label="Bereiche">
          {navigation.map((item) => (
            <a
              key={item.id}
              href={`#/${item.id}`}
              className={page === item.id ? "active" : ""}
              aria-current={page === item.id ? "page" : undefined}
              title={item.title}
              onClick={(event) => {
                event.preventDefault();
                navigate(item.id);
              }}
            >
              <item.icon size={19} aria-hidden="true" />
              <span className="nav-label">{item.title}</span>
            </a>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-card">
            <HardDrive size={18} aria-hidden="true" />
            <b>Made to stay local.</b>
            <p>
              Deine Daten.
              <br />
              Deine GPU. Dein Flow.
            </p>
            <span role="status">
              <span className={`status-dot ${connection}`} aria-hidden="true" />
              {connectionText[connection]}
            </span>
          </div>
          <button
            type="button"
            className="help-button"
            onClick={() => help.current?.showModal()}
          >
            <CircleHelp size={17} aria-hidden="true" /> So funktioniert’s
          </button>
          <div className="sidebar-version">
            Bob Studio <span>v0.1, experimentell</span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div>
            Workspace <span aria-hidden="true">/</span>
            <b>{navigation.find((item) => item.id === page)?.title}</b>
          </div>
          <div className="topbar-status">
            {running.length > 0 && (
              <button
                type="button"
                className="topbar-activity"
                onClick={scrollToActivity}
              >
                <LoaderCircle size={14} className="spin" aria-hidden="true" />
                {running.length === 1
                  ? `${jobTitle(running[0])}: ${statusLabel(running[0].status)}`
                  : `${running.length} Aufgaben aktiv`}
                <b>{Math.round(progress * 100)} %</b>
              </button>
            )}
            <span
              className="topbar-device"
              title={connectionText[connection]}
            >
              <span className={`status-dot ${connection}`} aria-hidden="true" />
              {hardware?.name || "Local engine"}
            </span>
          </div>
        </header>
        <main>
          {props ? (
            <>
              {page === "studio" && (
                <Studio
                  {...props}
                  goLibrary={() => navigate("library")}
                  shownOutput={shownOutput}
                />
              )}
              {page === "library" && <Library {...props} />}
              {page === "storage" && <Storage {...props} />}
              {page === "training" && (
                <Training {...props} goLibrary={() => navigate("library")} />
              )}
              {page === "models" && (
                <Models
                  onDeleted={onDeleted}
                  refresh={refresh}
                  onError={onError}
                  state={props.state}
                  goTraining={() => navigate("training")}
                />
              )}
              <section
                className="panel activity-panel"
                id="activity"
                tabIndex={-1}
                aria-labelledby="activity-title"
              >
                <div className="panel-heading">
                  <h2 id="activity-title">
                    <Sparkles size={16} aria-hidden="true" /> Aktivität
                  </h2>
                  <span className="tag">{running.length} aktiv</span>
                </div>
                <JobActivity jobs={props.state.jobs} onError={onError} />
              </section>
            </>
          ) : (
            <div className="large-empty" aria-busy={connection !== "offline"}>
              <AudioLines
                className={connection === "offline" ? "" : "spin"}
                size={40}
                aria-hidden="true"
              />
              <h2>
                {connection === "offline"
                  ? "Backend nicht erreichbar"
                  : "Studio wird verbunden …"}
              </h2>
              <p>Das lokale Backend muss auf Port 8000 laufen.</p>
              <button
                type="button"
                className="secondary"
                disabled={retrying}
                aria-busy={retrying}
                onClick={() => {
                  setRetrying(true);
                  refresh();
                }}
              >
                {retrying ? (
                  <LoaderCircle size={16} className="spin" aria-hidden="true" />
                ) : (
                  <RefreshCw size={16} aria-hidden="true" />
                )}
                {retrying ? "Wird verbunden …" : "Erneut verbinden"}
              </button>
            </div>
          )}
          <footer>
            <span>Built around your rhythm.</span>
            <span className="footer-tags">
              <span>osu!standard</span>
              <span>Local AI</span>
              <span>No cloud required</span>
            </span>
          </footer>
        </main>
      </div>
      <dialog
        ref={help}
        className="help-modal"
        aria-labelledby="help-title"
        onClick={(event) => {
          if (event.target === event.currentTarget) help.current?.close();
        }}
      >
        <div className="panel-heading">
          <h2 id="help-title">Dein Weg zur ersten Beatmap</h2>
          <button
            type="button"
            className="icon-button"
            aria-label="Schließen"
            onClick={() => help.current?.close()}
          >
            <X size={18} />
          </button>
        </div>
        <ol>
          <li>
            <b>Library:</b> osu!-Datenordner einlesen oder .osz importieren.
          </li>
          <li>
            <b>Auswählen:</b> Passende Maps als Trainingsdatensatz übernehmen.
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
      </dialog>
    </div>
  );
}
