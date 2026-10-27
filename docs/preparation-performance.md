# Song-Vorbereitung und Leerlauf · 27. Oktober 2026

Die Vorbereitung und das Status-Polling wurden lokal gemessen und optimiert. Keine Modellgewichte, Lernziele, Audioauflösung oder Generierungsregeln wurden geändert. Nutzer-Audios wurden nur gelesen; die Messungen verwendeten eine separate Datenbank und Caches unter `data/benchmarks/preparation-idle`.

## Vorbereitung

Acht vorhandene Maps mit unterschiedlichen Audio-Hashes, gleicher lokaler Rechner (16 logische CPU-Kerne). Vorher und nachher identische Dateien. Die Audiobibliothek wurde vor den Vergleichsläufen initialisiert, anschließend jeweils nur die Analyse-Caches des isolierten Benchmarks entfernt. Unprofilierte Messung:

| Zustand | Vorher | Nachher |
| --- | ---: | ---: |
| Ohne Analyse-Cache | 4,77 s | 3,40 s |
| Mit Analyse-Cache | 1,40 s | 0,065 s |
| Erneut mit Analyse-Cache | 1,40 s | 0,070 s |

Das entspricht in dieser kleinen Stichprobe etwa 29 % weniger Zeit ohne Cache und ungefähr Faktor 20 mit vollständigem Cache. Keine Hochrechnung auf 740 Songs oder Zusage für andere Hardware. Eine separate Profilierung zeigte beim ersten Start nach komplett geleertem Bibliothekscache etwa 21 Sekunden Import-/JIT-Aufwand; dieser einmalige Anteil ist in der Tabelle auf beiden Seiten ausgeschlossen und bleibt möglich.

Änderungen:

- Standardmäßig zwei CPU-Worker, gruppiert nach Audio-Hash; dadurch keine konkurrierenden Schreibzugriffe auf dasselbe Audio-Cachefile. Die erste Gruppe initialisiert die Audiobibliothek seriell. Maximal vier Worker, über `BOB_PREPARE_WORKERS` einstellbar.
- Die Reihenfolge der Trainingsmaps, Songgruppen, Ausschlüsse und spätere Trainingssplits bleiben erhalten. Abbruch wird auch in den Workern geprüft; beim Beenden werden wartende Aufgaben verworfen und laufende Worker abgewartet.
- Audio-Merkmale und Songzusammenfassungen werden pro Audio-Hash einmal je Vorbereitung geladen/berechnet.
- Bewegungsziele und Zeitraster können direkt aus einem vollständigen Cache geladen werden. Ein versioniertes Manifest prüft Frameanzahl, Formen und SHA-256 der Arrays. Fehlende oder beschädigte Dateien lösen Neuberechnung aus. Bei Änderungen an der Zieldefinition muss die Manifestversion erhöht werden.
- Beim Einlesen eines Sliders wird der geometrische Pfad einmal statt zweimal berechnet. Endpunkt und Kurvenanpassung verwenden denselben unveränderten Pfad.

Alle verglichenen Arrays (Mel-Audio, Events/Bewegungsziele, Frame-Labels, Bedingungen und Songzusammenfassung) waren vor/nach der Änderung **bitgenau identisch**. Das ist ein Nachweis unveränderter Vorbereitung für diese Stichprobe, kein neuer Nachweis der allgemeinen musikalischen Ergebnisqualität. Zusätzlich geprüft: Cache-Beschädigung, fehlende Manifeste, gemeinsame Audios, parallele Reihenfolge/Ausschlüsse/Abbruch und bestehende Parser-/Trainings-/Generierungstests.

Das standardmäßige 1-GiB-Limit wird weiterhin nach Aufgaben eingehalten. Bei größeren Korpora passt nicht alles hinein; entsprechend fällt der Vorteil beim nächsten Lauf kleiner aus. Die Speicherverwaltung erlaubt bewusst die Abwägung zwischen Platz und Wiederverwendung. Ein Cache-Limit von 0 verhindert den Geschwindigkeitsgewinn zwischen Aufgaben.

## Leerlauf

Vorher forderte die WebUI unabhängig vom Aufgabenstatus alle 2,5 Sekunden den vollständigen Zustand an und setzte anschließend neuen React-Zustand. Im geprüften Bestand waren das ungefähr 97 kB pro Antwort, einschließlich Modell-/Aufgabeninformationen. Das kann periodische CPU-Arbeit verursachen; eine eindeutige Zuordnung sämtlicher vom Nutzer beobachteten systemweiten CPU-Spitzen ist damit nicht bewiesen.

Jetzt:

- Sichtbarer, untätiger Tab: alle 30 Sekunden; bei Aufgaben: weiterhin alle 2,5 Sekunden.
- Versteckte Tabs: kein periodisches Polling; Fokus/Rückkehr und eigene Aktionen aktualisieren sofort.
- Eine durch Datenbank-Trigger gepflegte Revision plus Aufgaben-/Upload-Sperrstatus ermöglicht HTTP 304 ohne Aufbau des vollständigen Zustands. Löschungen und direkte SQL-Änderungen invalidieren die Revision ebenfalls.
- Eigene ETags mit deaktiviertem Browser-Response-Cache, damit 304 tatsächlich im Client ankommt. Bei 304 wird React-Zustand nicht neu gesetzt. Es läuft höchstens eine Statusanfrage gleichzeitig; nach Änderungen während einer Anfrage wird die alte Antwort verworfen und erneut abgefragt.
- Die Audio-Vorschau startet ihre Animationsschleife schon bisher nur während der Wiedergabe. Die Speicherinventur wird nicht regelmäßig im Leerlauf wiederholt.

Vergleich im selben TestClient-Prozess, jeweils 40 Anfragen: vollständiger Zustand 24,2 ms CPU / 25,2 ms Laufzeit und 97.101 Bytes; unverändert 4,3 ms CPU / 3,6 ms Laufzeit und 0 Body-Bytes. Diese Prozesswerte enthalten TestClient-Aufwand und sind keine Messung der gesamten Windows-CPU-Auslastung. Die ursprüngliche direkte Zustandsfunktion lag separat bei etwa 14,6 ms CPU ohne HTTP-TestClient.

Echte Browserprüfung mit isolierter App: erster Status 200, nächster Status nach 30,015 Sekunden 304; Hardware wurde nur beim Start abgefragt. Drei Frontend-Tests prüfen Intervalle, Sichtbarkeit, ausbleibende Render-Updates und konkurrierende Aktualisierungen. Backend-Tests sichern Revisionen bei Insert/Update/Delete und Uploads ab. Produktionsbuild erfolgreich.

Rohberichte: `data/benchmarks/preparation-idle/comparison.json`, `idle-comparison.json`, `browser-requests.json`. Verbliebene Benchmark-Dateien sind über die Speicherverwaltung entfernbar.
