import { useState } from "react";
import { AudioLines, ArrowUpRight, Sparkles, Upload } from "lucide-react";
import { api, upload } from "../api";
import { Slider } from "../components/Controls";
import { StudioAdvanced, defaultAdvanced } from "./StudioAdvanced";
import { MapPreview } from "../components/MapPreview";
import type { Device, Hardware, State } from "../types";
import { GeneratedMaps } from "./GeneratedMaps";
import { generationHelp } from "./generationHelp";

export function Studio({
  state,
  hardware,
  refresh,
  onError,
  goLibrary,
  onDeleted,
}: {
  state: State;
  hardware: Hardware | null;
  refresh: () => void;
  onError: (message: string) => void;
  goLibrary: () => void;
  onDeleted: (kind: "model" | "output", identifier: string) => void;
}) {
  const [model, setModel] = useState("");
  const [audio, setAudio] = useState<{ id: string; name: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [title, setTitle] = useState("");
  const [artist, setArtist] = useState("Unknown");
  const [version, setVersion] = useState("AI draft");
  const [device, setDevice] = useState<Device>(state.device);
  const [params, setParams] = useState({
    ar: 9,
    cs: 4,
    od: 7,
    hp: 5,
    density: 3,
    slider_bias: 0.5,
    spacing: 0.3,
    flow: 0.5,
  });
  const [advanced, setAdvanced] = useState(defaultAdvanced);
  const [previewId, setPreviewId] = useState("");
  const outputs = state.outputs.filter((output) => output.objects);
  const selectedModel =
    state.models.find((item) => item.id === model)?.id ||
    state.models[0]?.id ||
    "";
  const deletionBlocked = state.deletion_blocked;
  const selectedOutput =
    outputs.find((output) => output.id === previewId) || outputs[0];

  async function selectAudio(file?: File) {
    if (!file) return;
    setBusy(true);
    try {
      const value = await upload<{ id: string; name: string }>("/audio", file);
      setAudio(value);
      setTitle(file.name.replace(/\.[^.]+$/, ""));
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function generate() {
    setBusy(true);
    try {
      await api("/generate", {
        model_id: selectedModel,
        audio_id: audio?.id,
        title,
        artist,
        version,
        device,
        ...params,
        ...advanced,
        bpm: advanced.bpm ? Number(advanced.bpm) : null,
      });
      refresh();
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">DEIN SOUND. DEIN SPIELFELD.</div>
          <h1>
            From beat to <em>beatmap.</em>
          </h1>
          <p>Verwandle Audio mit deinem eigenen Modell in osu!-Beatmaps.</p>
        </div>
        <span className="pill">
          <span className="status-dot" /> Local-first studio
        </span>
      </div>
      {!state.models.length && (
        <div className="onboarding">
          <div className="step-number">01</div>
          <div>
            <b>Dein Mapping-Stil beginnt mit deiner Library.</b>
            <p>
              Importiere Maps, stelle Trainingsdaten zusammen und trainiere dein
              erstes Modell.
            </p>
          </div>
          <button className="text-button" onClick={goLibrary}>
            Library öffnen <ArrowUpRight size={17} />
          </button>
        </div>
      )}
      <div className="studio-grid">
        <div className="studio-settings">
          <section className="panel">
            <div className="panel-heading">
              <h2>
                <AudioLines size={18} /> Dein Track
              </h2>
              <span className="section-number">01</span>
            </div>
            <label
              className={`dropzone ${audio ? "loaded" : ""}`}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                void selectAudio(event.dataTransfer.files[0]);
              }}
            >
              <input
                type="file"
                accept=".mp3,.wav,.ogg,.flac,.m4a,.aac,.opus"
                onChange={(event) => void selectAudio(event.target.files?.[0])}
                disabled={busy}
              />
              <div className="upload-icon">
                <Upload size={22} />
              </div>
              <b>
                {busy
                  ? "Wird vorbereitet …"
                  : audio?.name || "Audio hier ablegen"}
              </b>
              <p>
                oder <span>Datei auswählen</span>
              </p>
              <small>MP3, WAV, OGG, FLAC & mehr · bis 512 MB</small>
            </label>
            <div className="field-row">
              <label>
                Titel
                <input
                  value={title}
                  onChange={(event) => setTitle(event.target.value)}
                  placeholder="Songtitel"
                />
              </label>
              <label>
                Artist
                <input
                  value={artist}
                  onChange={(event) => setArtist(event.target.value)}
                />
              </label>
            </div>
            <label>
              Mapping-Modell
              <select
                value={selectedModel}
                onChange={(event) => setModel(event.target.value)}
              >
                <option value="" disabled>
                  Trainiere zuerst ein Modell
                </option>
                {state.models.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
          </section>
          <section className="panel">
            <div className="panel-heading">
              <h2>
                <Sparkles size={17} /> Dein Mapping-Stil
              </h2>
              <span className="section-number">02</span>
            </div>
            <div className="preset-row">
              {[
                { name: "Easy", ar: 5, od: 3, density: 1.5 },
                { name: "Normal", ar: 7, od: 5, density: 2.5 },
                { name: "Hard", ar: 9, od: 7, density: 4 },
                { name: "Insane", ar: 9.5, od: 9, density: 6 },
              ].map((preset) => (
                <button
                  key={preset.name}
                  className={params.od === preset.od ? "selected" : ""}
                  onClick={() =>
                    setParams({
                      ...params,
                      ar: preset.ar,
                      od: preset.od,
                      density: preset.density,
                    })
                  }
                >
                  {preset.name}
                </button>
              ))}
            </div>
            <div className="slider-grid">
              {(["ar", "cs", "od", "hp"] as const).map((key) => (
                <Slider
                  key={key}
                  help={generationHelp[key]}
                  label={
                    {
                      ar: "Approach rate",
                      cs: "Circle size",
                      od: "Overall difficulty",
                      hp: "HP drain",
                    }[key]
                  }
                  value={params[key]}
                  onChange={(value) => setParams({ ...params, [key]: value })}
                />
              ))}
            </div>
            <Slider
              label="Objektdichte · pro Sekunde"
              help={generationHelp.density}
              value={params.density}
              min={0.5}
              onChange={(value) => setParams({ ...params, density: value })}
            />
            <div className="slider-grid">
              <Slider
                label="Bewegungsweite"
                help={generationHelp.spacing}
                value={params.spacing}
                min={0}
                max={1}
                step={0.05}
                onChange={(spacing) => setParams({ ...params, spacing })}
              />
              <Slider
                label="Richtungsfluss"
                help={generationHelp.flow}
                value={params.flow}
                min={0}
                max={1}
                step={0.05}
                onChange={(flow) => setParams({ ...params, flow })}
              />
            </div>
            <p className="fine-print">
              Weite und Richtungsfluss steuern den gelernten Stil: niedriger
              Fluss bevorzugt Umkehrungen, höherer Fluss fortlaufende
              Bewegungen. Keine feste Qualitätsbewertung.
            </p>
            <Slider
              label="Slider-Anteil · Stilvorgabe"
              help={generationHelp.sliders}
              value={params.slider_bias}
              min={0}
              max={1}
              step={0.05}
              onChange={(slider_bias) => setParams({ ...params, slider_bias })}
              hint="0: nur Circles. 1: Slider bevorzugen; sehr kurze Abstände können trotzdem Circles erfordern."
            />
            <StudioAdvanced
              value={advanced}
              onChange={setAdvanced}
              device={device}
              setDevice={setDevice}
              hardware={hardware}
              version={version}
              setVersion={setVersion}
            />
            <button
              className="primary generate-button"
              disabled={busy || !audio || !selectedModel || !title.trim()}
              onClick={() => void generate()}
            >
              <Sparkles size={18} /> Beatmap generieren{" "}
              <ArrowUpRight size={18} />
            </button>
            <p className="fine-print">
              Erstellt einen Beatmap-Entwurf. Presets sind keine garantierten
              Sternebewertungen.
            </p>
          </section>
        </div>
        <div className="studio-preview">
          <MapPreview output={selectedOutput} onError={onError} />
          <div className="workflow-strip">
            <span>
              <b>01</b> Audio analysieren
            </span>
            <i />
            <span>
              <b>02</b> Muster lernen
            </span>
            <i />
            <span>
              <b>03</b> Losspielen
            </span>
          </div>
          <GeneratedMaps
            outputs={outputs}
            selectedId={selectedOutput?.id}
            blocked={deletionBlocked}
            onSelect={setPreviewId}
            onDeleted={(identifier) => {
              if (identifier === selectedOutput?.id) setPreviewId("");
              onDeleted("output", identifier);
            }}
          />
          <div className="note">
            <span>✦</span>
            <p>
              <b>Dein Modell. Dein Stil.</b>
              <br />
              Alles bleibt lokal: deine Songs, deine Trainingsdaten und deine
              Modelle.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
