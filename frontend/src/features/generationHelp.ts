export const generationHelp = {
  ar: "Approach Rate (AR): Wie früh Objekte vor ihrem Trefferzeitpunkt erscheinen. Höher bedeutet weniger Vorbereitungszeit. Wird in der Beatmap gespeichert und dem Modell als Stilvorgabe gegeben.",
  cs: "Circle Size (CS): Größe der Trefferkreise. Ein höherer Wert macht die Kreise kleiner und das Zielen anspruchsvoller. Beeinflusst außerdem die Randabstände und Lesbarkeitsprüfung bei der Generierung.",
  od: "Overall Difficulty (OD): Wie genau du den vorgesehenen Zeitpunkt treffen musst. Höher bedeutet engere Zeitfenster für hohe Trefferwertungen. Beeinflusst das Star Rating nur wenig. Der Wert dient zusätzlich als Modellvorgabe.",
  hp: "HP Drain (HP): Wie streng die Lebensenergie bewertet wird. Höhere Werte machen es in der Regel schwerer, Fehler auszugleichen. Wird in der Map gespeichert und als Modellvorgabe genutzt; steuert nicht direkt die Notenanzahl.",
  density:
    "Gewünschte durchschnittliche Anzahl von Objektstarts pro Sekunde in hörbaren Abschnitten. Ein Slider zählt als ein Objekt, seine Ticks nicht einzeln. Das Modell verteilt dichtere und ruhigere Stellen; Auswahlregeln begrenzen die Menge. Keine Garantie für eine bestimmte Sternezahl.",
  spacing:
    "Stilvorgabe für Abstände zwischen aufeinanderfolgenden Objekten: niedriger eher kompakt, höher eher weite Bewegungen. Keine feste Pixelentfernung. Die Wirkung hängt von den gelernten Maps und dem verfügbaren Zeitabstand ab.",
  flow: "Stilvorgabe für aufeinanderfolgende Bewegungsrichtungen. Niedrigere Werte stehen eher für Umkehrungen, höhere eher für Bewegungen in ähnlicher Richtung. Kein Regler für Spielqualität; die Wirkung hängt vom Modell ab.",
  sliders:
    "Gewünschter Slider-Stil. Zwischen 0 und 1 entscheidet das Modell über Circle oder Slider; 0,5 garantiert deshalb nicht 50 Prozent Slider. 0 erzeugt nur Circles, 1 versucht Slider zu bevorzugen. Zu kurze Abstände oder ungeeignete Kurven können trotzdem Circles ergeben.",
  rhythm:
    "Wie frei das Modell zwischen passenden Rhythmen wählt (1/2-Sprünge, Triples, Streams, Pausen). Niedrig bleibt beim wahrscheinlichsten Rhythmus, höher wechselt öfter. Die Musik-Onsets fließen immer mit ein.",
  rhythmMode:
    "Patterns setzt Triples, Bursts und Streams als zusammenhängende Folgen auf das Beat-Raster. Onsets wählt zuerst alle Zeitpunkte aus den erkannten Musik-Onsets und plant danach Objekte, Slider und Vorausschau.",
  densityGuidance:
    "Wie stark der Rhythmus der vom Modell gelernten lokalen Notendichte folgt. Verhindert, dass ein einmal begonnener Stream bis zum Songende weiterläuft, und lässt in ruhigen Abschnitten Pausen zu. 0 = aus (altes Verhalten).",
  dynamics:
    "Wie stark leise Stellen ruhiger gemappt werden. Bei 1 halbiert sich die Zieldichte pro 6 dB, die ein Abschnitt leiser ist als die lauten Teile des Songs. 0 = Lautstärke ignorieren.",
  temperature:
    "Vielfalt der Positions- und Kurvenkandidaten. Niedrig bleibt näher an den wahrscheinlichsten Modellvorschlägen; höher erlaubt stärkere Varianten. Ändert nicht direkt die Objektdichte oder die Treffergenauigkeit. Mehr Vielfalt bedeutet nicht automatisch bessere Maps.",
};
