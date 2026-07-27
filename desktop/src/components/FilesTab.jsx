import { useEffect, useState } from "react";

function resultRows(payload) {
  return Array.isArray(payload?.results) ? payload.results : [];
}

export default function FilesTab({ request }) {
  const [roots, setRoots] = useState([]);
  const [root, setRoot] = useState("");
  const [path, setPath] = useState("");
  const [query, setQuery] = useState("");
  const [entries, setEntries] = useState([]);
  const [results, setResults] = useState([]);
  const [content, setContent] = useState("");
  const [status, setStatus] = useState("Loading allowed roots...");

  useEffect(() => {
    let active = true;
    request("GET", "/filesystem/roots")
      .then((payload) => {
        if (!active) return;
        const nextRoots = Array.isArray(payload?.allowed_roots) ? payload.allowed_roots : [];
        setRoots(nextRoots);
        setRoot(nextRoots[0] || "");
        setPath(nextRoots[0] || "");
        setStatus(nextRoots.length ? "Ready for bounded local search." : "No allowed filesystem roots are configured.");
      })
      .catch((error) => {
        if (active) setStatus(`Could not load filesystem roots: ${error.message || error}`);
      });
    return () => {
      active = false;
    };
  }, []);

  const listDirectory = async () => {
    try {
      const payload = await request("GET", `/filesystem/list?path=${encodeURIComponent(path)}`);
      setEntries(Array.isArray(payload?.entries) ? payload.entries : []);
      setResults([]);
      setContent("");
      setStatus(`Listed ${Number(payload?.entry_count || 0)} item(s).`);
    } catch (error) {
      setStatus(`List failed: ${error.message || error}`);
    }
  };

  const readFile = async () => {
    try {
      const payload = await request("GET", `/filesystem/read?path=${encodeURIComponent(path)}&max_bytes=65536`);
      setContent(String(payload?.text || ""));
      setEntries([]);
      setResults([]);
      setStatus(payload?.truncated ? "Read the first 65,536 bytes." : "File read complete.");
    } catch (error) {
      setStatus(`Read failed: ${error.message || error}`);
    }
  };

  const searchNames = async () => {
    try {
      const payload = await request(
        "GET",
        `/filesystem/search?root=${encodeURIComponent(root)}&q=${encodeURIComponent(query)}&max_results=50`
      );
      setResults(resultRows(payload));
      setEntries([]);
      setContent("");
      setStatus(`Filename search found ${resultRows(payload).length} result(s).`);
    } catch (error) {
      setStatus(`Filename search failed: ${error.message || error}`);
    }
  };

  const searchContent = async () => {
    try {
      const payload = await request("POST", "/filesystem/search_content", {
        root,
        q: query,
        max_results: 50,
        max_files: 500,
        max_bytes_per_file: 65536
      });
      setResults(resultRows(payload));
      setEntries([]);
      setContent("");
      setStatus(`Content search found ${resultRows(payload).length} result(s).`);
    } catch (error) {
      setStatus(`Content search failed: ${error.message || error}`);
    }
  };

  return (
    <section className="grid" data-testid="local-files-search">
      <div className="card">
        <h2>Files / Local Search</h2>
        <p className="help-text">
          Read-only access is limited to the roots shown here. Sensitive files, symlink escapes, and paths outside those roots are blocked.
        </p>
        <label>
          Allowed root
          <select value={root} onChange={(event) => {
            setRoot(event.target.value);
            setPath(event.target.value);
          }}>
            {roots.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <label>
          File or directory path
          <input value={path} onChange={(event) => setPath(event.target.value)} placeholder="Choose a path inside an allowed root" />
        </label>
        <div className="row-actions">
          <button disabled={!path} onClick={listDirectory} type="button">List directory</button>
          <button disabled={!path} onClick={readFile} type="button">Read text file</button>
        </div>
        <label>
          Search query
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filename or text to find" />
        </label>
        <div className="row-actions">
          <button disabled={!root || !query.trim()} onClick={searchNames} type="button">Search filenames</button>
          <button disabled={!root || !query.trim()} onClick={searchContent} type="button">Search file contents</button>
        </div>
        <p className="status-line">{status}</p>
      </div>

      <div className="card">
        <h2>Results</h2>
        <div className="model-list">
          {entries.map((entry) => (
            <button
              className="history-entry"
              key={`${entry.type}:${entry.name}`}
              onClick={() => setPath(`${path.replace(/\/$/, "")}/${entry.name}`)}
              type="button"
            >
              <strong>{entry.name}</strong> <span>{entry.type}</span>
            </button>
          ))}
          {results.map((item) => (
            <button className="history-entry" key={item.path} onClick={() => setPath(item.path)} type="button">
              <strong>{item.path}</strong>{item.snippet ? <span>{item.snippet}</span> : null}
            </button>
          ))}
          {content ? <pre className="filesystem-read-preview">{content}</pre> : null}
          {!entries.length && !results.length && !content ? <p className="empty">No local file result selected yet.</p> : null}
        </div>
      </div>
    </section>
  );
}
