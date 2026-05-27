import { useEffect, useMemo, useState } from "react";
import { Download, Search, Loader2, CheckCircle, XCircle, FileText, Image, RotateCcw, FolderOpen, AlertTriangle, ArrowUp, ArrowDown, ArrowUpDown } from "lucide-react";
import { api } from "../api/client";
import type {
  HfDownloadEvent, HfDownloadPlan, HfRepoListing, HfSearchResult, ModelInfo,
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
  // Modelos locais — usado pra marcar "já baixado" em resultados de busca e
  // no picker de quant. Detecção é por prefixo `owner/repo[/subdir]` do
  // relative_path (que o backend gera a partir das pastas configuradas).
  models: ModelInfo[];
  onStart:  (req: { repo_id: string; rel_paths: string[]; subdir?: string | null; base_dir: string }) => Promise<void>;
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
  downloadId, plan, events, progress, done, error, models,
  onStart, onCancel, onReset,
}: Props) {
  const [url, setUrl]               = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<HfSearchResult[] | null>(null);
  const [searching, setSearching]   = useState(false);
  // Sort dos resultados da busca: backend já devolve por downloads desc;
  // colunas clicáveis alternam asc/desc.
  const [sortKey, setSortKey] = useState<"repo_id" | "downloads" | "likes">("downloads");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  // Conjunto de prefixos `owner/repo/...` (lowercase) presentes localmente.
  // O backend salva downloads em {root}/{owner}/{repo}/{subdir}/{file}, então
  // os relative_path dos ModelInfo carregam esse prefixo. Mesmo se o usuário
  // baixou via outra ferramenta e a árvore estiver levemente diferente, o
  // match no nível repo só dá falso negativo (mostra "baixe" pra repo que
  // já tem), nunca falso positivo.
  const downloadedPrefixes = useMemo(() => {
    const s = new Set<string>();
    for (const m of models) s.add(m.relative_path.replace(/\\/g, "/").toLowerCase());
    return s;
  }, [models]);

  const isRepoDownloaded = (repoId: string) => {
    const pref = repoId.toLowerCase() + "/";
    for (const rp of downloadedPrefixes) if (rp.startsWith(pref)) return true;
    return false;
  };

  const isQuantDownloaded = (repoId: string, quant: string) => {
    const pref = `${repoId.toLowerCase()}/${quant.toLowerCase()}/`;
    for (const rp of downloadedPrefixes) if (rp.startsWith(pref)) return true;
    return false;
  };

  const sortedResults = useMemo(() => {
    if (!searchResults) return null;
    const arr = [...searchResults];
    arr.sort((a, b) => {
      let cmp: number;
      if (sortKey === "repo_id") cmp = a.repo_id.localeCompare(b.repo_id);
      else cmp = a[sortKey] - b[sortKey];
      return sortDir === "asc" ? cmp : -cmp;
    });
    return arr;
  }, [searchResults, sortKey, sortDir]);

  const toggleSort = (key: "repo_id" | "downloads" | "likes") => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      // Default: texto → asc, números → desc (downloads/likes maiores primeiro)
      setSortDir(key === "repo_id" ? "asc" : "desc");
    }
  };

  const [listing, setListing]       = useState<HfRepoListing | null>(null);
  const [listingLoading, setListingLoading] = useState(false);
  const [chosenQuant, setChosenQuant] = useState<string | null>(null);
  const [includeMmproj, setIncludeMmproj] = useState(true);
  const [pickerError, setPickerError] = useState<string | null>(null);

  const [configuredPaths, setConfiguredPaths] = useState<string[]>([]);
  const [chosenPath, setChosenPath] = useState<string | null>(null);
  const [pathsLoading, setPathsLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const s = await api.getSettings();
        setConfiguredPaths(s.model_paths);
        if (s.model_paths.length > 0) {
          setChosenPath((cur) => cur && s.model_paths.includes(cur) ? cur : s.model_paths[0]);
        } else {
          setChosenPath(null);
        }
      } catch {
        setConfiguredPaths([]);
      } finally {
        setPathsLoading(false);
      }
    })();
  }, []);

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
      // Auto-select só se tiver 1 quant E ainda não estiver baixado;
      // caso contrário deixa o usuário escolher conscientemente.
      if (keys.length === 1 && !isQuantDownloaded(l.repo_id, keys[0])) {
        setChosenQuant(keys[0]);
      }
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
    if (!listing || !chosenQuant || !chosenPath) return;
    // Guard final: backend já pula arquivos existentes (file_skip), mas
    // bloquear aqui evita ruído desnecessário no log de download.
    if (isQuantDownloaded(listing.repo_id, chosenQuant)) return;
    const files = listing.quants[chosenQuant] ?? [];
    const rels = files.map((f) => f.path);
    if (includeMmproj) {
      for (const m of listing.mmprojs) rels.push(m.path);
    }
    await onStart({
      repo_id: listing.repo_id,
      rel_paths: rels,
      subdir: chosenQuant,
      base_dir: chosenPath,
    });
  }

  const noPathsConfigured = !pathsLoading && configuredPaths.length === 0;

  const downloadActive = downloadId !== null && !done;
  const showProgressSection = !!plan;

  return (
    <div className="space-y-4">
      {/* ── Destino do download (pasta cadastrada) ───────────────────── */}
      <section className="bg-ink-900 border border-ink-800 rounded-lg p-4 space-y-2">
        <h2 className="font-medium text-ink-100 flex items-center gap-2 text-sm">
          <FolderOpen className="w-4 h-4" /> Salvar em
        </h2>
        {pathsLoading ? (
          <div className="text-xs text-ink-500 flex items-center gap-1">
            <Loader2 className="w-3 h-3 animate-spin" /> carregando…
          </div>
        ) : noPathsConfigured ? (
          <div className="text-sm bg-amber-950/40 border border-amber-800 text-amber-200 rounded p-2 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>
              Nenhuma pasta cadastrada. Vá em <span className="mono">Settings</span> e adicione pelo menos um caminho antes de baixar.
            </span>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {configuredPaths.map((p) => (
              <label
                key={p}
                className={`border rounded p-2 cursor-pointer text-sm flex items-center gap-2 ${
                  chosenPath === p
                    ? "border-accent bg-accent/10"
                    : "border-ink-700 hover:border-ink-500"
                }`}
              >
                <input
                  type="radio"
                  className="accent-accent"
                  checked={chosenPath === p}
                  onChange={() => setChosenPath(p)}
                />
                <span className="mono truncate" title={p}>{p}</span>
              </label>
            ))}
          </div>
        )}
      </section>

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
      {sortedResults && (
        <section className="bg-ink-900 border border-ink-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-ink-200 mb-2">
            {sortedResults.length} repositório(s) GGUF encontrados
          </h3>
          <div className="max-h-64 overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-ink-900 border-b border-ink-800 text-xs text-ink-400">
                <tr>
                  <SortHeader label="Repositório" align="left"  k="repo_id"   sortKey={sortKey} sortDir={sortDir} onClick={toggleSort} />
                  <SortHeader label="Downloads"   align="right" k="downloads" sortKey={sortKey} sortDir={sortDir} onClick={toggleSort} />
                  <SortHeader label="Likes"       align="right" k="likes"     sortKey={sortKey} sortDir={sortDir} onClick={toggleSort} />
                </tr>
              </thead>
              <tbody>
                {sortedResults.map((r) => {
                  const dl = isRepoDownloaded(r.repo_id);
                  return (
                    <tr
                      key={r.repo_id}
                      onClick={() => resolveAndList(r.repo_id)}
                      className="cursor-pointer hover:bg-ink-800/60 border-b border-ink-800/50 last:border-b-0"
                      title={dl ? "já baixado (clique pra ver outros quants)" : undefined}
                    >
                      <td className="px-3 py-1.5 mono truncate max-w-0 w-full">
                        <span className={dl ? "text-ink-400" : "text-ink-100"}>{r.repo_id}</span>
                        {dl && (
                          <span className="ml-2 inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/60 text-emerald-300 align-middle">
                            <CheckCircle className="w-3 h-3" /> baixado
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-1.5 mono tabular-nums text-ink-300 text-right whitespace-nowrap">
                        {r.downloads.toLocaleString()}
                      </td>
                      <td className="px-3 py-1.5 mono tabular-nums text-ink-300 text-right whitespace-nowrap">
                        {r.likes.toLocaleString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
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
              {Object.entries(listing.quants).sort(([a], [b]) => a.localeCompare(b)).map(([k, files]) => {
                const dl = isQuantDownloaded(listing.repo_id, k);
                return (
                  <label
                    key={k}
                    title={dl ? "este quant já está baixado localmente" : undefined}
                    className={`border rounded p-2 text-sm ${
                      dl
                        ? "border-ink-800 bg-ink-950/40 opacity-60 cursor-not-allowed"
                        : chosenQuant === k
                          ? "border-accent bg-accent/10 cursor-pointer"
                          : "border-ink-700 hover:border-ink-500 cursor-pointer"
                    }`}
                  >
                    <input
                      type="radio"
                      className="hidden"
                      checked={chosenQuant === k}
                      disabled={dl}
                      onChange={() => { if (!dl) setChosenQuant(k); }}
                    />
                    <div className="mono font-medium truncate flex items-center gap-1.5">
                      {k}
                      {dl && <CheckCircle className="w-3 h-3 text-emerald-400 shrink-0" />}
                    </div>
                    <div className="text-xs text-ink-500">
                      {files.length} arquivo(s) · {(files.reduce((s, f) => s + (f.size || 0), 0) / (1024 ** 3)).toFixed(2)} GB
                      {dl && <span className="text-emerald-400 ml-1">· baixado</span>}
                    </div>
                  </label>
                );
              })}
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

          {(() => {
            const chosenDownloaded = !!(listing && chosenQuant && isQuantDownloaded(listing.repo_id, chosenQuant));
            return (
              <button
                disabled={!chosenQuant || downloadActive || !chosenPath || chosenDownloaded}
                onClick={startDownload}
                className="px-3 py-1.5 rounded bg-accent hover:bg-accent/80 text-white text-sm disabled:opacity-50 flex items-center gap-1"
                title={
                  chosenDownloaded ? "este quant já está baixado"
                  : downloadActive ? "já há um download em curso"
                  : !chosenPath ? "cadastre uma pasta em Settings primeiro"
                  : ""
                }
              >
                <Download className="w-3 h-3" /> baixar
              </button>
            );
          })()}
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

type SortKey = "repo_id" | "downloads" | "likes";

function SortHeader({
  label, align, k, sortKey, sortDir, onClick,
}: {
  label: string;
  align: "left" | "right";
  k: SortKey;
  sortKey: SortKey;
  sortDir: "asc" | "desc";
  onClick: (k: SortKey) => void;
}) {
  const active = sortKey === k;
  const Icon = !active ? ArrowUpDown : sortDir === "asc" ? ArrowUp : ArrowDown;
  return (
    <th
      onClick={() => onClick(k)}
      className={`px-3 py-1.5 select-none cursor-pointer font-normal hover:text-ink-200 ${
        align === "right" ? "text-right" : "text-left"
      } ${active ? "text-ink-100" : ""}`}
    >
      <span className={`inline-flex items-center gap-1 ${align === "right" ? "flex-row-reverse" : ""}`}>
        {label}
        <Icon className={`w-3 h-3 ${active ? "" : "opacity-40"}`} />
      </span>
    </th>
  );
}
