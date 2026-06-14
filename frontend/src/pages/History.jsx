import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { clearHistory, readHistory } from "../lib/history";

function formatDate(value) {
    if (!value) return "–";
    return new Date(value).toLocaleString();
}

function History({ currentUser }) {
    const navigate = useNavigate();
    const [history, setHistory] = useState(() => readHistory(currentUser));

    useEffect(() => {
        setHistory(readHistory(currentUser));
    }, [currentUser]);

    function handleClearHistory() {
        clearHistory(currentUser);
        setHistory([]);
    }

    return (
        <main className="history">
            <section className="purple-page purple-page--top history-hero">
                <div className="history-card card card--lg">
                    <h1 className="history-card__title">
                        History of the analyzed compositions
                    </h1>
                    <p className="history-card__subtitle">
                        {currentUser
                            ? `Saved for profile: ${currentUser.login}`
                            : "Guest history. Sign in to keep a separate profile history."}
                    </p>

                    <div className="history-table-wrapper">
                        <table className="history-table">
                            <thead>
                            <tr>
                                <th scope="col">Composition</th>
                                <th scope="col">Genre</th>
                                <th scope="col">Date of analysis</th>
                            </tr>
                            </thead>
                            <tbody>
                            {history.length === 0 ? (
                                <tr>
                                    <td colSpan={3} style={{ opacity: 0.8 }}>
                                        No items yet. Analyze a track first.
                                    </td>
                                </tr>
                            ) : (
                                history.map((row) => (
                                    <tr
                                        key={row.id || `${row.filename}-${row.timestamp}`}
                                        className="history-table__row"
                                        onClick={() => navigate("/result", {
                                            state: {
                                                data: row.data || { filename: row.filename, predictions: row.predictions },
                                                fileName: row.filename,
                                                fromHistory: true,
                                            },
                                        })}
                                    >
                                        <td>{row.filename}</td>
                                        <td>
                                            {row.genre}
                                            {row.confidence ? ` (${row.confidence})` : ""}
                                        </td>
                                        <td>{formatDate(row.timestamp)}</td>
                                    </tr>
                                ))
                            )}
                            </tbody>
                        </table>
                    </div>

                    <div className="history-card__actions" style={{ gap: 12 }}>
                        <button
                            type="button"
                            className="btn btn--dark history-card__btn"
                            onClick={() => navigate("/")}
                        >
                            Return to homepage
                        </button>
                        <button
                            type="button"
                            className="btn btn--light history-card__btn"
                            onClick={handleClearHistory}
                            disabled={history.length === 0}
                        >
                            Clear history
                        </button>
                    </div>
                </div>
            </section>
        </main>
    );
}

export default History;
