import React, { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { predict } from "../lib/api";

function Analyzing() {
    const location = useLocation();
    const navigate = useNavigate();

    const file = location.state?.file || null;
    const fileName = useMemo(() => {
        if (location.state?.fileName) return location.state.fileName;
        if (file?.name) return file.name;
        return "Unknown audio file";
    }, [location.state, file]);

    const [error, setError] = useState("");

    useEffect(() => {
        if (!file) {
            navigate("/analyze", { replace: true });
            return;
        }

        const controller = new AbortController();

        (async () => {
            try {
                setError("");
                const startedAt = Date.now();
                const data = await predict(file, { signal: controller.signal });
                const elapsed = Date.now() - startedAt;
                if (elapsed < 1800) {
                    await new Promise((resolve) => setTimeout(resolve, 1800 - elapsed));
                }
                navigate("/result", {
                    replace: true,
                    state: {
                        fileName: data.filename || fileName,
                        data,
                    },
                });
            } catch (err) {
                if (controller.signal.aborted) return;
                setError(err?.message || "Prediction failed");
            }
        })();

        return () => controller.abort();
    }, [file, fileName, navigate]);

    return (
        <main className="analyzing">
            <section className="purple-page analyzing-hero">
                <div className="analyzing-card card card--md" aria-live="polite">
                    <div className="analyzing-card__header">
                        <div className="analyzing-spinner" aria-hidden="true" />
                        <div>
                            <h1 className="analyzing-card__title">
                                {error ? "Analysis failed" : "We analyze your composition..."}
                            </h1>
                            <p className="analyzing-card__file">
                                File: <span>{fileName}</span>
                            </p>
                            {error && (
                                <div style={{ marginTop: 12 }}>
                                    <p style={{ color: "#ffb4b4" }}>{error}</p>
                                    <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
                                        <button
                                            type="button"
                                            className="btn btn--dark"
                                            onClick={() => navigate("/analyze")}
                                        >
                                            Back
                                        </button>
                                        <button
                                            type="button"
                                            className="btn btn--primary"
                                            onClick={() => navigate(0)}
                                        >
                                            Retry
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </section>
        </main>
    );
}

export default Analyzing;
