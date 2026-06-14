import { getHistoryKey } from "./auth";

export function readHistory(user) {
    try {
        const value = JSON.parse(localStorage.getItem(getHistoryKey(user)) || "[]");
        return Array.isArray(value) ? value : [];
    } catch {
        return [];
    }
}

export function saveHistoryItem(user, item) {
    const key = getHistoryKey(user);
    const previous = readHistory(user);
    const next = [item, ...previous].slice(0, 50);
    localStorage.setItem(key, JSON.stringify(next));
}

export function clearHistory(user) {
    localStorage.removeItem(getHistoryKey(user));
}
