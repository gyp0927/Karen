// Thin fetch wrapper for the Flask REST surface. All endpoints are routed
// through Vite's proxy in dev and same-origin in production.

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  const res = await fetch(input, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `HTTP ${res.status}`);
  }
  const ct = res.headers.get("content-type") ?? "";
  return ct.includes("application/json") ? ((await res.json()) as T) : ((await res.text()) as unknown as T);
}

export const api = {
  get: <T>(p: string) => request<T>(p),
  post: <T>(p: string, body?: unknown) =>
    request<T>(p, { method: "POST", body: body == null ? undefined : JSON.stringify(body) }),
  del: <T>(p: string) => request<T>(p, { method: "DELETE" }),
  upload: async <T>(p: string, file: File, extra?: Record<string, string>): Promise<T> => {
    const form = new FormData();
    form.append("file", file);
    if (extra) for (const [k, v] of Object.entries(extra)) form.append(k, v);
    const res = await fetch(p, { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text().catch(() => `HTTP ${res.status}`));
    return (await res.json()) as T;
  },
};
