import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { loginUser } from "../lib/auth";

function Login({ onLogin }) {
    const navigate = useNavigate();
    const [form, setForm] = useState({
        login: "",
        password: "",
    });
    const [error, setError] = useState("");

    function onChange(e) {
        const { name, value } = e.target;
        setForm((previous) => ({
            ...previous,
            [name]: value,
        }));
    }

    function handleSubmit(e) {
        e.preventDefault();
        try {
            const user = loginUser(form.login, form.password);
            onLogin(user);
            navigate("/");
        } catch (err) {
            setError(err?.message || "Login failed.");
        }
    }

    return (
        <main className="login">
            <section className="purple-page login-hero">
                <div className="login-card card card--lg">
                    <h1 className="login-card__title">Login to the system</h1>

                    <form className="login-form" onSubmit={handleSubmit}>
                        {error && <p className="form-error">{error}</p>}

                        <div className="login-form__group">
                            <label htmlFor="login-username" className="login-form__label">
                                Login or e-mail:
                            </label>
                            <input
                                id="login-username"
                                name="login"
                                type="text"
                                className="login-form__input"
                                placeholder="enter login or email"
                                value={form.login}
                                onChange={onChange}
                                required
                            />
                        </div>

                        <div className="login-form__group">
                            <label htmlFor="login-password" className="login-form__label">
                                Password:
                            </label>
                            <input
                                id="login-password"
                                name="password"
                                type="password"
                                className="login-form__input"
                                placeholder="enter password"
                                value={form.password}
                                onChange={onChange}
                                required
                            />
                        </div>

                        <div className="login-form__actions">
                            <button type="submit" className="btn btn--dark login-form__btn">
                                Log In
                            </button>

                            <button
                                type="button"
                                className="btn btn--dark login-form__btn"
                                onClick={() => navigate("/register")}
                            >
                                Register
                            </button>

                            <button
                                type="button"
                                className="btn btn--dark login-form__btn"
                                onClick={() => alert("Password reset flow – demo")}
                            >
                                Forgot password?
                            </button>
                        </div>
                    </form>
                </div>
            </section>
        </main>
    );
}

export default Login;
