// src/pages/Register.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { registerUser } from "../lib/auth";

function Register({ onRegister }) {
    const navigate = useNavigate();

    const [form, setForm] = useState({
        login: "",
        email: "",
        password: "",
        accepted: false,
    });
    const [error, setError] = useState("");

    function onChange(e) {
        const { name, value, type, checked } = e.target;
        setForm((prev) => ({
            ...prev,
            [name]: type === "checkbox" ? checked : value,
        }));
    }

    function handleSubmit(e) {
        e.preventDefault();
        setError("");

        if (!form.accepted) {
            setError("Please accept the terms of use.");
            return;
        }

        if (form.password.length < 4) {
            setError("Password must contain at least 4 characters.");
            return;
        }

        try {
            const user = registerUser(form);
            onRegister(user);
            navigate("/");
        } catch (err) {
            setError(err?.message || "Registration failed.");
        }
    }

    return (
        <main className="register">
            <section className="purple-page register-hero" aria-labelledby="reg-title">
                <div className="register-card card card--lg">
                    <h1 id="reg-title" className="register-card__title">
                        Registration
                    </h1>

                    <p className="register-card__subtitle">
                        Create an account and save your history of analyzed songs in one place.
                    </p>

                    <form className="register-form" onSubmit={handleSubmit}>
                        {error && <p className="form-error">{error}</p>}

                        <div className="register-form__group">
                            <label htmlFor="reg-login" className="register-form__label">
                                Your login:
                            </label>
                            <input
                                id="reg-login"
                                name="login"
                                type="text"
                                className="register-form__input"
                                placeholder="enter your login"
                                value={form.login}
                                onChange={onChange}
                                required
                            />
                        </div>

                        <div className="register-form__group">
                            <label htmlFor="reg-email" className="register-form__label">
                                E-mail:
                            </label>
                            <input
                                id="reg-email"
                                name="email"
                                type="email"
                                className="register-form__input"
                                placeholder="enter your email address"
                                value={form.email}
                                onChange={onChange}
                                required
                            />
                        </div>

                        <div className="register-form__group">
                            <label htmlFor="reg-password" className="register-form__label">
                                Password:
                            </label>
                            <input
                                id="reg-password"
                                name="password"
                                type="password"
                                className="register-form__input"
                                placeholder="enter your password"
                                value={form.password}
                                onChange={onChange}
                                required
                            />
                        </div>

                        <div className="register-form__group register-form__group--checkbox">
                            <input
                                id="reg-terms"
                                name="accepted"
                                type="checkbox"
                                className="register-form__checkbox"
                                checked={form.accepted}
                                onChange={onChange}
                                required
                            />
                            <label htmlFor="reg-terms" className="register-form__checkbox-label">
                                I agree to the processing of personal data and the terms of use.
                            </label>
                        </div>

                        <div className="register-form__actions">
                            <button type="submit" className="btn btn--dark register-form__btn">
                                Register
                            </button>

                            <button
                                type="button"
                                className="btn btn--dark register-form__btn"
                                onClick={() => navigate("/login")}
                            >
                                Already have an account? Log in
                            </button>
                        </div>
                    </form>
                </div>
            </section>
        </main>
    );
}

export default Register;
