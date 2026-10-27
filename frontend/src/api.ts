export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
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

async function readResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Server nicht erreichbar" }));
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail),
    );
  }
  return response.json();
}

export async function upload<T>(path: string, file: File): Promise<T> {
  const data = new FormData();
  data.append("file", file);
  return readResponse<T>(
    await fetch(`/api${path}`, { method: "POST", body: data }),
  );
}

export async function deleteArtifact(
  kind: "model" | "output",
  identifier: string,
): Promise<void> {
  await readResponse(
    await fetch(
      `/api/${kind === "model" ? "models" : "outputs"}/${encodeURIComponent(identifier)}`,
      { method: "DELETE" },
    ),
  );
}
