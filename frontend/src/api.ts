const OFFLINE =
  "Server nicht erreichbar. Läuft das lokale Backend auf Port 8000?";

async function send(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new Error(OFFLINE);
  }
}

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await send(
    `/api${path}`,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  return readResponse<T>(response);
}

function detail(error: unknown, fallback: string) {
  const value = (error as { detail?: unknown } | null)?.detail;
  if (typeof value === "string") return value;
  if (Array.isArray(value))
    return value
      .map((entry) =>
        typeof entry?.msg === "string" ? entry.msg : JSON.stringify(entry),
      )
      .join(" ");
  return value === undefined ? fallback : JSON.stringify(value);
}

async function readResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(
      detail(
        error,
        response.status >= 500 ? OFFLINE : `Anfrage fehlgeschlagen (${response.status})`,
      ),
    );
  }
  return response.json();
}

/** Uploads a file. With `onProgress`, reports the sent fraction from 0 to 1. */
export async function upload<T>(
  path: string,
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<T> {
  const data = new FormData();
  data.append("file", file);
  if (!onProgress)
    return readResponse<T>(
      await send(`/api${path}`, { method: "POST", body: data }),
    );
  return new Promise<T>((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `/api${path}`);
    request.responseType = "text";
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onerror = () => reject(new Error(OFFLINE));
    request.onload = () => {
      let parsed: unknown = null;
      try {
        parsed = JSON.parse(request.responseText);
      } catch {
        /* Non-JSON answer; handled below. */
      }
      if (request.status >= 200 && request.status < 300) {
        onProgress(1);
        resolve(parsed as T);
      } else
        reject(
          new Error(
            detail(
              parsed,
              request.status >= 500
                ? OFFLINE
                : `Upload fehlgeschlagen (${request.status})`,
            ),
          ),
        );
    };
    request.send(data);
  });
}

export async function remove(path: string): Promise<void> {
  await readResponse(await send(`/api${path}`, { method: "DELETE" }));
}

export async function deleteArtifact(
  kind: "model" | "output",
  identifier: string,
): Promise<void> {
  await remove(
    `/${kind === "model" ? "models" : "outputs"}/${encodeURIComponent(identifier)}`,
  );
}

/** Accepts a file when its extension is in the list (".mp3", …). */
export function acceptsFile(file: File, extensions: string[]) {
  const name = file.name.toLowerCase();
  return extensions.some((extension) => name.endsWith(extension));
}
