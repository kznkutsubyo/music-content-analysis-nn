const USERS_KEY = "genreUsers";
const CURRENT_USER_KEY = "genreCurrentUser";

function safeRead(key, fallback) {
    try {
        const value = JSON.parse(localStorage.getItem(key) || "null");
        return value ?? fallback;
    } catch {
        return fallback;
    }
}

function writeUsers(users) {
    localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

export function getUsers() {
    const users = safeRead(USERS_KEY, []);
    return Array.isArray(users) ? users : [];
}

export function getCurrentUser() {
    const currentId = localStorage.getItem(CURRENT_USER_KEY);
    if (!currentId) return null;
    return getUsers().find((user) => user.id === currentId) || null;
}

export function registerUser({ login, email, password }) {
    const normalizedLogin = login.trim();
    const normalizedEmail = email.trim().toLowerCase();
    const users = getUsers();

    if (users.some((user) => user.login.toLowerCase() === normalizedLogin.toLowerCase())) {
        throw new Error("This login is already registered.");
    }

    if (users.some((user) => user.email.toLowerCase() === normalizedEmail)) {
        throw new Error("This e-mail is already registered.");
    }

    const user = {
        id: typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`,
        login: normalizedLogin,
        email: normalizedEmail,
        password,
        createdAt: new Date().toISOString(),
    };

    writeUsers([...users, user]);
    localStorage.setItem(CURRENT_USER_KEY, user.id);
    return user;
}

export function loginUser(loginOrEmail, password) {
    const value = loginOrEmail.trim().toLowerCase();
    const user = getUsers().find((item) => (
        item.login.toLowerCase() === value || item.email.toLowerCase() === value
    ));

    if (!user || user.password !== password) {
        throw new Error("Invalid login or password.");
    }

    localStorage.setItem(CURRENT_USER_KEY, user.id);
    return user;
}

export function logoutUser() {
    localStorage.removeItem(CURRENT_USER_KEY);
}

export function getHistoryKey(user) {
    return user ? `genreHistory:${user.id}` : "genreHistory:guest";
}
