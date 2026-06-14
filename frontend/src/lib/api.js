export const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001";

export async function predict(file, opts = {}) {
    if (!file) {
        throw new Error("No file selected");
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
        signal: opts.signal,
    });

    if (!response.ok) {
        const body = await response.json().catch(() => null);
        const message = body?.detail || response.statusText || "Prediction failed";
        throw new Error(message);
    }

    return response.json();
}

export async function fetchMetrics(opts = {}) {
    const response = await fetch(`${API_BASE}/metrics`, {
        signal: opts.signal,
    });

    if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || response.statusText || "Failed to load metrics");
    }

    return response.json();
}
