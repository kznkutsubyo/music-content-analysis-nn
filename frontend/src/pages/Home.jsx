import React from "react";
import { useNavigate } from "react-router-dom";

function Home() {
    const navigate = useNavigate();

    return (
        <main className="home">
            <section
                className="purple-page home-hero"
                aria-labelledby="hero-title"
            >
                <div className="home-hero__inner">
                    <h1 id="hero-title" className="home-hero__title">
                        AI Music Genre Analyzer
                    </h1>

                    <p className="home-hero__subtitle">Analyze your music by AI</p>

                    <p className="home-hero__description">
                        Upload a song, find out its musical genre, and see how confident
                        the neural network is.
                    </p>

                    <div className="home-hero__actions">
                        <button
                            type="button"
                            className="btn btn--primary home-hero__btn"
                            onClick={() => navigate("/analyze")}
                        >
                            Upload a song
                        </button>

                        <button
                            type="button"
                            className="btn btn--dark home-hero__btn"
                            onClick={() => navigate("/history")}
                        >
                            History of the analyzed compositions
                        </button>
                    </div>
                </div>
            </section>
        </main>
    );
}

export default Home;
