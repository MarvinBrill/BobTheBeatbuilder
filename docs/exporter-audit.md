# Exporter-Prüfung

Geprüft am 26.10.2026: `kabiiQ/BeatmapExporter`, Commit `718acc03c9a63232be6c943eb87e1212e3454b88`.

## Umfang und Befunde

Fokussierte Quellcodeprüfung von Projekt-/Builddateien, Realm-Zugriff, Hashpfaden, zentralen Exportfunktionen, Updateprüfung, Prozessstarts und auffindbaren Netzwerk-/Dateischreiboperationen. Kein vollständiges Audit aller transitiven Abhängigkeiten oder Release-Binaries.

**Kein offensichtlicher Schadcode im geprüften Umfang gefunden.** Beobachtet wurden:

- `ExporterUpdater`: Versionsabfrage einer GitHub-Datei, keine automatische Installation in der untersuchten Methode.
- `PlatformUtil.Open`: Shell-gestützter Programmstart für Ordner und Links.
- `Transcoder`: FFmpeg-Aufruf über FFMpegCore.
- Buildskripte: .NET-Veröffentlichung, Paketierung und Löschen eigener Build-Ausgaben.
- Breite Exportfunktionen einschließlich Schreibzugriffen und Zusammenführen bestehender Collections.

Diese Funktionen werden nicht übernommen. Bob referenziert keine Projekte des geklonten Exporters und führt dessen Skripte oder Programme nicht aus.

## Eigene Implementierung

Nur Strukturkenntnisse wurden verwendet: `client.realm`, `BeatmapSet.Files`, `Filename`, `File.Hash` und `files/<Zeichen 1>/<Zeichen 1–2>/<Hash>`. Hashdateien sind normalerweise nicht verschlüsselt; Realm liefert die Namenszuordnung.

Der neue C#-Leser hat eine direkte Abhängigkeit: Realm 20.1.0 von NuGet. Keine Netzwerkaufrufe, Updateprüfung, Shell-Starts oder Extraktion. Er liest eine Kopie aus `data/snapshots` dynamisch und schreibgeschützt. Die Schemaversion wird gegebenenfalls aus der Realm-Fehlermeldung übernommen; keine Migration der Quelle.

Vor/nach der Kopie werden Größe und Änderungszeit verglichen. Ein aktiver Client kann trotzdem Inkonsistenzen verursachen; osu! vor dem Scan schließen. Neue Realm-Dateiformate können ein Reader-Update erfordern. Fehler führen nicht zum Öffnen der Quelle mit Schreibrechten.

## Schutzmaßnahmen

- Container-Spielordner schreibgeschützt, eigene Daten unter `data`.
- Hashvalidierung, Pfadauflösung innerhalb des Zielordners, kein unbeschränktes `extractall`.
- Limits für Uploads, entpackte Archive, Mitgliederzahl und `.osu`-Textgröße.
- Nur `.osu` und Audio aus Trainingsarchiven; keine ausführbaren Dateien.
- Beschädigte Dateien, insbesondere Nullbyte-Inhalte, werden abgewiesen.
- FFmpeg ohne Shell, mit Argumentliste, Zeitlimit und ohne Netzwerkprotokolle.
- Modellimport prüft Safetensors, Architektur, Formen, Datentypen und endliche Werte. Kein `torch.load`/Pickle.
- Keine Übermittlung von Songs, Trainingsdaten oder Modellen an externe Dienste.

## Bezugsquellen und Speicher

Für den Sternefilter liest der eigene Reader zusätzlich `BeatmapSet.Beatmaps` und ordnet `BeatmapInfo.StarRating` über den Map-Hash zu. Nur endliche, nicht negative Werte werden übernommen; der offizielle Unbekannt-Wert `-1` bleibt unbekannt. Grundlage: [offizielles BeatmapInfo-Schema](https://github.com/ppy/osu/blob/master/osu.Game/Beatmaps/BeatmapInfo.cs). Die Bewertung stammt aus dem lokalen lazer-Cache, es wird kein zusätzlicher fremder Difficulty-Rechner ausgeführt. Die Quelldatenbank wird weiterhin ausschließlich über eine private Kopie gelesen.

Python-Pakete: PyPI, PyTorch: offizieller Index `download.pytorch.org`, DirectML: Microsoft-Paket auf PyPI. npm verwendet einen Lockfile mit Integritätswerten, Realm kommt von NuGet. Container nutzen offizielle Node- und Microsoft-.NET-Images sowie Ubuntu-Paketquellen.

Keine fremden Release-Executables, Modellgewichte oder Installationsskripte wurden heruntergeladen und ausgeführt. Toolchains, FFmpeg und CUDA-PyTorch waren bereits vorhanden. Neue Pakete/Caches liegen unter `data`; Programme auf C: werden lediglich verwendet. Der erste .NET-Aufruf meldete seine übliche Entwicklerzertifikat-Initialisierung; Setup und Start deaktivieren sie künftig ausdrücklich. Docker-Images wurden wegen des auf C: liegenden Docker-Speichers nicht gebaut.

Eine Registry und feste direkte Paketversionen sind kein vollständiger Vertrauensnachweis für sämtliche transitiven Abhängigkeiten. Die Prüfung garantiert keine allgemeine Schadcodefreiheit.
