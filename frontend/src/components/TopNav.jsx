import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import logo from "../assets/logo.svg";

function TopNav({ currentUser, theme, onLogout, onToggleTheme }) {
    const navigate = useNavigate();
    const linkClass = ({ isActive }) =>
        isActive ? "topnav__link topnav__link--active" : "topnav__link";

    return (
        <header className="topnav">
            <div className="topnav__inner">
                <div className="topnav__logo">
                    <img className="topnav__logo-icon" src={logo} alt="" aria-hidden="true" />
                    <span className="visually-hidden">AI Music Genre Analyzer</span>
                </div>

                <nav className="topnav__nav" aria-label="Main navigation">
                    <NavLink to="/analyze" className={linkClass}>
                        Analyze
                    </NavLink>
                    <NavLink to="/metrics" className={linkClass}>
                        Metrics
                    </NavLink>
                    <a href="#community" className="topnav__link">
                        Community
                    </a>
                    <a href="#contact" className="topnav__link">
                        Contact
                    </a>
                </nav>

                <div className="topnav__auth">
                    <button
                        type="button"
                        className="theme-switch"
                        role="switch"
                        aria-checked={theme === "dark"}
                        aria-label="Toggle dark theme"
                        onClick={onToggleTheme}
                    >
                        <span className="theme-switch__track">
                            <span className="theme-switch__icon theme-switch__icon--sun" aria-hidden="true" />
                            <span className="theme-switch__icon theme-switch__icon--moon" aria-hidden="true" />
                            <span className="theme-switch__thumb" />
                        </span>
                    </button>

                    {currentUser ? (
                        <>
                            <span className="topnav__user">{currentUser.login}</span>
                            <button
                                type="button"
                                className="btn btn--ghost topnav__auth-btn"
                                onClick={onLogout}
                            >
                                Log out
                            </button>
                        </>
                    ) : (
                        <>
                            <button
                                type="button"
                                className="btn btn--ghost topnav__auth-btn"
                                onClick={() => navigate("/login")}
                            >
                                Sign in
                            </button>

                            <button
                                type="button"
                                className="btn btn--primary topnav__auth-btn"
                                onClick={() => navigate("/register")}
                            >
                                Register
                            </button>
                        </>
                    )}
                </div>
            </div>
        </header>
    );
}

export default TopNav;
