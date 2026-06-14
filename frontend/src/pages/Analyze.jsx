import React, { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

function Analyze() {
    const [file, setFile] = useState(null);
    const [fileName, setFileName] = useState("");
    const [isDragging, setIsDragging] = useState(false);
    const [error, setError] = useState("");
    const fileInputRef = useRef(null);
    const navigate = useNavigate();

    function setSelectedFile(selectedFile) {
        setError("");

        if (selectedFile && selectedFile.size > 20 * 1024 * 1024) {
            setError("Maximum file size is 20 MB.");
            setFile(null);
            setFileName("");
            return;
        }

        setFile(selectedFile);
        setFileName(selectedFile ? selectedFile.name : "");
    }

    function handleFileChange(e) {
        const selectedFile = e.target.files?.[0] || null;
        setSelectedFile(selectedFile);
    }

    function handleDragOver(e) {
        e.preventDefault();
        setIsDragging(true);
    }

    function handleDragLeave(e) {
        e.preventDefault();
        setIsDragging(false);
    }

    function handleDrop(e) {
        e.preventDefault();
        setIsDragging(false);
        setSelectedFile(e.dataTransfer.files?.[0] || null);
    }

    function handleAnalyzeClick() {
        if (!file && fileInputRef.current) {
            fileInputRef.current.click();
            return;
        }
        if (!file) return;
        navigate("/analyzing", { state: { file, fileName: file.name } });
    }

    return (
        <main className="upload">
            <section
                className="purple-page upload-hero"
                aria-labelledby="upload-title"
            >
                <form className="upload-card card card--md" onSubmit={(e) => e.preventDefault()}>
                    <label
                        htmlFor="track-file"
                        className={`upload-card__dropzone ${isDragging ? "upload-card__dropzone--active" : ""}`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                    >
                        <div className="upload-card__icon" aria-hidden="true">↑</div>

                        <div>
                            <h1 id="upload-title" className="upload-card__title">
                                Upload a music track
                            </h1>

                            <p className="upload-card__text">
                                Drag the file here
                                <br />
                                or click to select from your computer
                            </p>

                            {fileName && (
                                <p className="upload-card__filename">
                                    Selected file: <strong>{fileName}</strong>
                                </p>
                            )}

                            {error && <p className="upload-card__error">{error}</p>}

                            <p className="upload-card__meta">
                                Supported formats: WAV, MP3, FLAC, OGG, AU, AIFF, M4A<br />
                                Maximum file size: 20 MB
                            </p>
                        </div>
                    </label>

                    <input
                        id="track-file"
                        ref={fileInputRef}
                        type="file"
                        accept=".wav,.mp3,.flac,.ogg,.oga,.au,.aiff,.aif,.m4a,.aac,audio/*"
                        className="visually-hidden"
                        onChange={handleFileChange}
                    />

                    <div className="upload-card__footer">
                        <button
                            type="button"
                            className="btn btn--primary upload-card__button"
                            onClick={handleAnalyzeClick}
                        >
                            Analyze
                        </button>
                    </div>
                </form>
            </section>
        </main>
    );
}

export default Analyze;
