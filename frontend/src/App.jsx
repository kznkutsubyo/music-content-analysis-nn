import React, { useEffect, useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";

import TopNav from "./components/TopNav.jsx";
import Home from "./pages/Home.jsx";
import Analyze from "./pages/Analyze.jsx";
import Analyzing from "./pages/Analyzing.jsx";
import Result from "./pages/Result.jsx";
import History from "./pages/History.jsx";
import Login from "./pages/Login.jsx";
import Register from "./pages/Register.jsx";
import Metrics from "./pages/Metrics.jsx";
import { getCurrentUser, logoutUser } from "./lib/auth";

function App() {
    const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "light");
    const [currentUser, setCurrentUser] = useState(() => getCurrentUser());

    useEffect(() => {
        document.documentElement.dataset.theme = theme;
        localStorage.setItem("theme", theme);
    }, [theme]);

    function toggleTheme() {
        setTheme((current) => (current === "dark" ? "light" : "dark"));
    }

    function handleLogout() {
        logoutUser();
        setCurrentUser(null);
    }

    return (
        <div className="app-root">
            <TopNav
                currentUser={currentUser}
                theme={theme}
                onLogout={handleLogout}
                onToggleTheme={toggleTheme}
            />
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/analyze" element={<Analyze />} />
                <Route path="/analyzing" element={<Analyzing />} />
                <Route path="/result" element={<Result currentUser={currentUser} />} />
                <Route path="/history" element={<History currentUser={currentUser} />} />
                <Route path="/metrics" element={<Metrics />} />
                <Route path="/login" element={<Login onLogin={setCurrentUser} />} />
                <Route path="/register" element={<Register onRegister={setCurrentUser} />} />
                <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
        </div>
    );
}

export default App;
