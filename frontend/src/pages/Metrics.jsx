import React, { useEffect, useState } from "react";
import { fetchMetrics } from "../lib/api";

function percent(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "–";
    return `${(value * 100).toFixed(1)}%`;
}

function ms(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "–";
    return `${value.toFixed(1)} ms`;
}

function Metrics() {
    const [methods, setMethods] = useState([]);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const controller = new AbortController();

        (async () => {
            try {
                setLoading(true);
                setError("");
                const data = await fetchMetrics({ signal: controller.signal });
                setMethods(Array.isArray(data.methods) ? data.methods : []);
            } catch (err) {
                if (!controller.signal.aborted) {
                    setError(err?.message || "Failed to load metrics.");
                }
            } finally {
                if (!controller.signal.aborted) {
                    setLoading(false);
                }
            }
        })();

        return () => controller.abort();
    }, []);

    return (
        <main className="metrics">
            <section className="purple-page purple-page--top metrics-hero">
                <div className="metrics-card card card--lg">
                    <h1 className="metrics-card__title">Model comparison</h1>
                    <p className="metrics-card__subtitle">
                        Test-set quality and inference speed for the implemented genre classifiers.
                    </p>

                    {loading && <p className="metrics-card__status">Loading metrics...</p>}
                    {error && <p className="form-error">{error}</p>}

                    {!loading && !error && (
                        <div className="metrics-grid">
                            {methods.map((method) => (
                                <article className="metrics-method" key={method.id}>
                                    <h2 className="metrics-method__title">{method.name}</h2>
                                    <dl className="metrics-method__list">
                                        <div>
                                            <dt>Accuracy</dt>
                                            <dd>{percent(method.metrics?.accuracy)}</dd>
                                        </div>
                                        <div>
                                            <dt>Top-3</dt>
                                            <dd>{percent(method.metrics?.top3_accuracy)}</dd>
                                        </div>
                                        <div>
                                            <dt>Macro F1</dt>
                                            <dd>{percent(method.metrics?.macro_f1)}</dd>
                                        </div>
                                        <div>
                                            <dt>Speed</dt>
                                            <dd>{ms(method.speed?.ms_per_item)}</dd>
                                        </div>
                                    </dl>
                                </article>
                            ))}
                        </div>
                    )}
                </div>
            </section>
        </main>
    );
}

export default Metrics;
