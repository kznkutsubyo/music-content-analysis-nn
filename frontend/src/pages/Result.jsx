// src/pages/Result.jsx
import React, { useEffect, useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { saveHistoryItem } from "../lib/history";

function toPercent(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "–";
    const percent = value * 100;
    if (percent > 0 && percent < 0.1) return "<0.1";
    return percent.toFixed(1);
}

function formatGenre(value) {
    if (!value) return "Unknown";
    return value
        .split(/([\s-]+)/)
        .map((part) => (/^[a-z]/.test(part) ? part.charAt(0).toUpperCase() + part.slice(1) : part))
        .join("");
}

function getPrimaryPrediction(predictions) {
    return predictions?.AST?.top1
        || Object.values(predictions || {}).find((method) => method?.top1)?.top1
        || null;
}

function getAlternativePredictions(predictions) {
    const astTopk = predictions?.AST?.topk;
    if (Array.isArray(astTopk) && astTopk.length > 1) {
        return astTopk.slice(1, 4);
    }

    return Object.entries(predictions || {})
        .map(([method, result]) => ({
            name: result?.top1?.label,
            prob: result?.top1?.prob,
            method,
        }))
        .filter((item) => item.name)
        .slice(0, 3);
}

function getMethodResults(predictions) {
    return [
        { key: "AST", name: "AST" },
        { key: "CNN", name: "CNN" },
        { key: "RandomForest", name: "Random Forest" },
        { key: "KNN", name: "KNN" },
    ]
        .map((method) => ({
            ...method,
            top1: predictions?.[method.key]?.top1 || null,
            topk: predictions?.[method.key]?.topk || [],
        }))
        .filter((method) => method.top1);
}

function getAgreement(methodResults) {
    if (methodResults.length < 2) return null;
    const labels = methodResults.map((method) => method.top1.label);
    const first = labels[0];
    return labels.every((label) => label === first);
}

function Result({ currentUser }) {
    const location = useLocation();
    const navigate = useNavigate();

    const data = location.state?.data || null;
    const predictions = data?.predictions || null;
    const fileName = location.state?.fileName || data?.filename || "Unknown file";
    const fromHistory = Boolean(location.state?.fromHistory);

    const primary = useMemo(() => getPrimaryPrediction(predictions), [predictions]);
    const alternatives = useMemo(() => getAlternativePredictions(predictions), [predictions]);
    const methodResults = useMemo(() => getMethodResults(predictions), [predictions]);
    const modelsAgree = useMemo(() => getAgreement(methodResults), [methodResults]);
    const mainGenre = formatGenre(primary?.label);
    const confidence = toPercent(primary?.prob);

    useEffect(() => {
        if (!data?.predictions || fromHistory) return;

        const item = {
            id: typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`,
            timestamp: new Date().toISOString(),
            filename: data.filename || fileName,
            genre: mainGenre,
            confidence: `${confidence}%`,
            data,
            predictions: data.predictions,
        };

        saveHistoryItem(currentUser, item);
    }, [currentUser, data, fileName, fromHistory, mainGenre, confidence]);

    if (!data || !predictions) {
        return (
            <main className="result">
                <section className="purple-page result-hero">
                    <div className="result-card card card--md">
                        <div className="result-card__header">
                            <div className="result-card__icon" aria-hidden="true">!</div>
                            <h1 className="result-card__title">No result available</h1>
                        </div>
                        <div className="result-card__actions">
                            <button
                                type="button"
                                className="btn btn--dark result-card__btn"
                                onClick={() => navigate("/analyze")}
                            >
                                Analyze a song
                            </button>
                        </div>
                    </div>
                </section>
            </main>
        );
    }

    return (
        <main className="result">
            <section className="purple-page result-hero">
                <div className="result-card card card--md">
                    <div className="result-card__header">
                        <div className="result-card__icon" aria-hidden="true">
                            ✓
                        </div>
                        <h1 className="result-card__title">Result of the analysis</h1>
                    </div>

                    <div className="result-card__content">
                        <p className="result-card__line">
                            <strong>Genre:</strong> {mainGenre}
                        </p>
                        <p className="result-card__line">
                            <strong>Confidence level:</strong> {confidence}%
                        </p>
                        {modelsAgree !== null && (
                            <p className={`result-card__agreement ${modelsAgree ? "is-agree" : "is-split"}`}>
                                {modelsAgree ? "All models agree" : "Models differ"}
                            </p>
                        )}

                        <div className="result-card__block">
                            <p className="result-card__line--label">
                                <strong>Alternative genres:</strong>
                            </p>
                            <ul className="result-card__list">
                                {alternatives.map((alt, index) => (
                                    <li key={`${alt.name || alt.label}-${index}`}>
                                        {formatGenre(alt.label || alt.name)} ({toPercent(alt.prob)}%)
                                    </li>
                                ))}
                            </ul>
                        </div>

                        <div className="result-card__block">
                            <p className="result-card__line--label">
                                <strong>Model outputs:</strong>
                            </p>
                            <div className="result-models">
                                {methodResults.map((method) => (
                                    <div className="result-model" key={method.key}>
                                        <div className="result-model__row">
                                            <span className="result-model__name">{method.name}</span>
                                            <span className="result-model__genre">{formatGenre(method.top1.label)}</span>
                                            <span className="result-model__prob">{toPercent(method.top1.prob)}%</span>
                                        </div>
                                        <div className="result-model__bar" aria-hidden="true">
                                            <span style={{ width: `${Math.max(2, Math.min(100, method.top1.prob * 100))}%` }} />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <p className="result-card__file">
                            File: <span>{fileName}</span>
                        </p>
                    </div>

                    <div className="result-card__actions">
                        <button
                            type="button"
                            className="btn btn--dark result-card__btn"
                            onClick={() => navigate("/analyze")}
                        >
                            Analyze the next song
                        </button>

                        <button
                            type="button"
                            className="btn btn--dark result-card__btn"
                            onClick={() => navigate("/history")}
                        >
                            History of the analyzed compositions
                        </button>
                    </div>

                    <button
                        type="button"
                        className="result-card__feedback"
                        onClick={() => alert("Thank you for your feedback!")}
                    >
                        I disagree with the result
                    </button>
                </div>
            </section>
        </main>
    );
}

export default Result;
