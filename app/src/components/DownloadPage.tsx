import { useState } from "react";
import { Download, Search, Loader2, CheckCircle, XCircle, FileText, Image, RotateCcw } from "lucide-react";
import { api } from "../api/client";
import type {
  HfDownloadEvent, HfDownloadPlan, HfRepoListing, HfSearchResult,
} from "../api/types";
import type { DownloadProgress } from "../App";

type Props = {
  // Estado de download elevado pro App — sobrevive a troca de aba.
  downloadId: string | null;
  plan: HfDownloadPlan | null;
  events: HfDownloadEvent[];
  progress: Record<string, DownloadProgress>;
  done: boolean;
  error: string | null;
  onStart:  (req: { repo_id: string; rel_paths: string[]; subdir?: string | null }) => Promise<void>;
  onCancel: () => Promise<void>;
  onReset:  () => void;
};

// Fluxo (estado local só do picker — URL/search/listing):
//   1. Usuário cola URL HF ou faz search.
//   2. Lista repos / quants disponíveis no repo.
//   3. Pick quant + (opcionalmente) mmproj.
//   4. Click "baixar" → App.handleStartDownload abre SSE no nível superior.
//   5. Trocar de aba não interrompe; mostra spinner na aba "Download".
export function DownloadPage({
  downloadId, plan, events, progress, done, error,
  onStart, onCancel, onReset,
}: Props) {
  const [url, setUrl]               = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<HfSearchResult[] | null>(null);
  const [searching, setSearching]   = useState(false);

  const [listing, setListing]       = useState<HfRepoListing | null>(null);
  const [listingLoading, setListingLoading] = useState(false);
  const [chosenQuant, setChosenQuant] = useState<string | null>(null);
  const [includeMmproj, setIncludeMmproj] = useState(true);
  const [pickerError, setPickerError] = useState<string | null>(null);

  async function resolveAndList(target: string) {
    setPickerError(null); setListing(null); setChosenQuant(null);
    setListingLoading(true);
    try {
      let repoId = target;
      if (target.startsWith("http")) {
        const r = await api.hfResolve(target);
        repoId = r.repo_id;
      }
      const l = await api.hfList(repoId);
      setListing(l);
      const keys = Object.keys(l.quants);
      if (keys.length === 1) setChosenQuant(keys[0]);
    } catch (e) {
      setPickerError((e as Error).message);
    } finally {
      setListingLoading(false);
    }
  }

  async function search() {
    if (!searchQuery.trim()) return;
    setPickerError(null); setSearching(true); setListing(null);
    try {
      const r = await api.hfSearch(searchQuery.trim(), 25);
      setSearchResults(r.results);
    } catch (e) {
      setPickerError((e as Error).message);
    } finally {
      setSearching(false);
    }
  }

  async function startDownload() {
    if (!listing || !chosenQuant) return;
    const files = listing.quants[chosenQuant] ?? [];
    const rels = files.map((f) => f.path);
    if (includeMmproj) {
      for (const m of listing.mmprojs) rels.push(m.path);
    }
    await onStart({
      repo_id: listing.repo_id,
      rel_paths: rels,
      subdir: chosenQuant,
    });
  }

  const downloadActive = downloadId !== null && !done;
  const showProgressSection = !!plan;

  return (
    <div className="space-y-4">
      {/* ── URL / Search ──────────────────────────────────────────────── */}
      <section className="bg-ink-900 border border-ink-800 rounded-lg p-4 space-y-3">
        <h2 className="font-medium text-ink-100 flex items-center gap-2">
          <Download className="w-4 h-4" /> HuggingFace download
        </h2>
        <div className="flex gap-2">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && url.trim() && resolveAndList(url.trim())}
            placeholder="https://huggingface.co/owner/repo  ou  owner/repo"
            className="flex-1 bg-ink-950 border border-ink-700 rounded px-3 py-1.5 text-sm focus:outline-none focus:border-accent mono"
          />
          <button
            disabled={!url.trim() || listingLoading}
            onClick={() => resolveAndList(url.trim())}
            className="px-3 py-1.5 rounded bg-accent hover:bg-accent/80 text-white text-sm disabled:opacity-50 flex items-center gap-1"
          >
            {listingLoading && <Loader2 className="w-3 h-3 animate-spin" />}
            inspecionar
          </button>
        </div>

        <div className="flex gap-2">
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && search()}
            placeholder="ou busca: qwen 30b moe coder…"
            className="flex-1 bg-ink-950 border border-ink-700 rounded px-3 py-1.5 text-sm focus:outline-none focus:border-accent"
          />
          <button
            disabled={!searchQuery.trim() || searching}
            onClick={search}
            className="px-3 py-1.5 rounded bg-ink-800 hover:bg-ink-700 text-sm disabled:opacity-50 flex items-center gap-1"
          >
            {searching ? <Loader2 className="w-3 h-3 animate-spin" /> : <Search className="w-3 h-3" />} buscar
          </button>
        </div>

        {pickerError && (
          <div className="text-sm text-red-400 mono">{pickerError}</div>
        )}
      </section>

      {/* ── Search results ───────────────────────────────────────────── */}
      {searchResults && (
        <section className="bg-ink-900 border border-ink-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-ink-200 mb-2">
            {searchResults.length} repositório(s) GGUF encontrados
          </h3>
          <ul className="text-sm space-y-1 max-h-64 overflow-auto">
            {searchResults.map((r) => (
              <li key={r.repo_id}>
                <button
                  onClick={() => resolveAndList(r.repo_id)}
                  className="w-full text-left px-3 py-1.5 rounded hover:bg-ink-800 flex items-center gap-3"
                >
                  <span className="mono text-ink-100 flex-1 truncate">{r.repo_id}</span>
                  <span className="text-xs text-ink-500 mono shrink-0">
                    ↓ {r.downloads.toLocaleString()} · ♥ {r.likes}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* ── Repo listing → quant picker ──────────────────────────────── */}
      {listing && (
        <section className="bg-ink-900 border border-ink-800 rounded-lg p-4 space-y-3">
          <h3 className="text-sm font-medium text-ink-200">
            <span className="mono text-accent-fg">{listing.repo_id}</span> — escolha um quant
          </h3>
          {Object.keys(listing.quants).length === 0 ? (
            <p className="text-sm text-amber-400">Nenhum .gguf neste repo.</p>
          ) : (
            <div className="grid grid-cols-2 gap-2 max-h-64 overflow-auto">
              {Object.entries(listing.quants).sort(([a], [b]) => a.localeCompare(b)).map(([k, files]) => (
                <label
                  key={k}
                  className={`border rounded p-2 cursor-pointer text-sm ${
                    chosenQuant === k
                      ? "border-accent bg-accent/10"
                      : "border-ink-700 hover:border-ink-500"
                  }`}
                >
                  <input
                    type="radio"
                    className="hidden"
                    checked={chosenQuant === k}
                    onChange={() => setChosenQuant(k)}
                  />
                  <div className="mono font-medium truncate">{k}</div>
                  <div className="text-xs text-ink-500">
                    {files.length} arquivo(s) · {(files.reduce((s, f) => s + (f.size || 0), 0) / (1024 ** 3)).toFixed(2)} GB
                  </div>
                </label>
              ))}
            </div>
          )}

          {listing.mmprojs.length > 0 && (
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={includeMmproj}
                onChange={(e) => setIncludeMmproj(e.target.checked)}
              />
              <Image className="w-3.5 h-3.5" />
              incluir mmproj ({listing.mmprojs.map((m) => m.path.split("/").pop()).join(", ")})
            </label>
          )}

          <button
            disabled={!chosenQuant || downloadActive}
            onClick={startDownload}
            className="px-3 py-1.5 rounded bg-accent hover:bg-accent/80 text-white text-sm disabled:opacity-50 flex items-center gap-1"
            title={downloadActive ? "já há um download em curso" : ""}
          >
            <Download className="w-3 h-3" /> baixar
          </button>
        </section>
      )}

      {/* ── Download progress (estado lifted) ────────────────────────── */}
      {showProgressSection && plan && (
        <section className="bg-ink-900 border border-ink-800 rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-ink-200 flex items-center gap-2">
              {downloadActive && <Loader2 className="w-3.5 h-3.5 animate-spin text-accent-fg" />}
              Progresso · destino: <span className="mono text-ink-400">{plan.base_dir}</span>
            </h3>
            <div className="flex items-center gap-2">
              {downloadActive && (
                <button
                  onClick={onCancel}
                  className="text-xs px-2 py-1 rounded bg-red-900/60 hover:bg-red-800 text-red-100 flex items-center gap-1"
                >
                  <XCircle className="w-3 h-3" />cancelar
                </button>
              )}
              {done && (
                <button
                  onClick={onReset}
                  className="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-ink-700 text-ink-300 flex items-center gap-1"
                  title="Limpar e baixar outro modelo"
                >
                  <RotateCcw className="w-3 h-3" />limpar
                </button>
              )}
            </div>
          </div>

          {error && (
            <div className="text-sm text-red-400 mono">{error}</div>
          )}

          <ul className="space-y-2">
            {plan.items.map((it) => {
              const pf = progress[it.rel];
              const skipped = events.some((e) => e.type === "file_skip" && e.rel === it.rel);
              const finished = pf?.done || skipped;
              const pct = pf?.total ? Math.min(100, (pf.dl / pf.total) * 100) : 0;
              return (
                <li key={it.rel} className="text-xs space-y-1">
                  <div className="flex items-center gap-2">
                    <FileText className="w-3 h-3 text-ink-500 shrink-0" />
                    <span className="mono flex-1 truncate" title={it.rel}>{it.rel.split("/").pop()}</span>
                    {skipped && <span className="text-ink-500 text-[11px]">já existe</span>}
                    {finished && !skipped && <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0" />}
                    {pf && !finished && (
                      <span className="mono tabular-nums text-ink-400 shrink-0">
                        {(pf.dl / (1024 ** 3)).toFixed(2)} / {(pf.total / (1024 ** 3)).toFixed(2)} GB
                        {" · "}{(pf.speed / (1024 ** 2)).toFixed(1)} MB/s
                      </span>
                    )}
                  </div>
                  <div className="h-1.5 bg-ink-800 rounded overflow-hidden">
                    <div
                      className={`h-full ${finished ? "bg-emerald-500" : "bg-accent"}`}
                      style={{ width: finished ? "100%" : `${pct}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>

          {done && !error && (
            <div className="text-sm text-emerald-400 flex items-center gap-1">
              <CheckCircle className="w-4 h-4" /> concluído. Os modelos já aparecem na lista.
            </div>
          )}
        </section>
      )}
    </div>
  );
}
