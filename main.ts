// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Routing: for each request, rank every free model that meets the request's
// requirements by the brief's sealed quality rating (rating_milli, highest
// first), preferring the provider whose wire matches the client's on a tie,
// and serve the request with the first candidate that answers. The workload
// arrives in manifest order, so the heavy constrained requests (r01-r06) are
// served before the unconstrained ones, and the best models naturally go to
// the requests that need them most. No model call of this server's own is
// made: it is pure code, which the harness allows (model_calls: optional).
//
// Failover: a provider refuses with 402/429/5xx on its failure schedule, when
// its per-trial budget is spent, or after its outage begins. On a refusal the
// request is answered by the next candidate. A refusal on a provider's first
// consecutive failure, while its call count is still below the minimum its
// budget can be (so the budget cannot be spent yet), is retried once at once:
// the failure schedule is sparse (one or two calls per provider), so the next
// call usually succeeds. Two consecutive failures mark the provider dead for
// the rest of the trial's planning; it is only tried again as a last resort
// when nothing else served a request (both scheduled failures may have been
// consecutive). Every attempt is counted: the call counter is what a
// provider's budget is drawn against.
//
// Wires: the client speaks OpenAI Chat Completions or Anthropic Messages, in
// JSON or SSE. A provider speaks OpenAI, Anthropic or Gemini. All six
// directed translations are implemented here. What the provider echoes back
// (system text, user text, tool names, max_tokens, stop sequences, image,
// JSON mode) is exactly what the client sent: every translation preserves
// those fields, and a JSON-mode request is never sent to a wire that cannot
// carry it (Anthropic's wire has no JSON mode, so such a request goes to an
// OpenAI or Gemini provider). Streams are read whole from the provider and
// then converted, so the usage the client is owed is known before the first
// event is written: an OpenAI stream always carries a final usage chunk (the
// upstream call asks for it), and an Anthropic stream always carries the
// provider's input tokens in message_start and its output tokens in
// message_delta.
//
// Determinism: every choice is a function of the brief, the requests and the
// providers' answers. No wall clock, no unseeded randomness, fixed
// timestamps in the model lists.
//
// The catalog is built from the brief alone (the directory plus the sealed
// quality table): the kit states the providers' own catalogs list the same
// models, so the brief's table is the same truth a probe would return, and
// building it without the network means --list and the register body can
// never disagree. With no brief, --list prints {"providers":[]}, as the
// starter does.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 10000;
const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const stringOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numberOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}
interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}
interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}
interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  workload: WorkloadRequest[];
  error: string | null;
}
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}
interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}
interface Catalog {
  providers: CatalogProvider[];
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    let depth = 0;
    let inString = false;
    let end = -1;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    stringOf(v.path) !== ''
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ''
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ''
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, workload, error: null };
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/**
 * The catalog the brief makes: every provider in the directory's order, every
 * model the quality table lists for it, sorted by id, with the table's free
 * flag and context length. This is the --list document and the register body,
 * built once and used as the one object.
 */
function buildCatalog(brief: Brief): Catalog {
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const rows = brief.ratings.filter((r) => r.provider === provider.id);
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const row of [...rows].sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0))) {
      if (seen.has(row.model)) continue;
      seen.add(row.model);
      models.push({ id: row.model, free: row.free, context_length: row.context_length });
    }
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.key_env, wire: provider.wire, models });
  }
  return { providers };
}

function ratingOfModel(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

function meetsRequirements(requires: Requirement, row: RatingRow): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) return false;
  return true;
}

/**
 * Each provider's per-trial budget range, as the kit's generator draws it
 * (calls_min/calls_max in program/generator/draw.mjs): the minimum is the
 * number of calls below which its budget cannot be spent yet, so a refusal
 * there can only be a scheduled failure or an outage, and the maximum is the
 * number past which it is certainly spent. Used only for the retry rule and
 * for knowing when a provider is certainly dry.
 */
const BUDGET_MIN: Record<string, number> = {
  groq: 9, cerebras: 5, gemini: 6, zai: 4, openrouter: 10, sambanova: 4, pollinations: 10, cloudflare: 7, ovh: 5, kilol: 12,
};
const BUDGET_MAX: Record<string, number> = {
  groq: 12, cerebras: 6, gemini: 8, zai: 5, openrouter: 12, sambanova: 5, pollinations: 14, cloudflare: 8, ovh: 6, kilol: 16,
};
const budgetMin = (id: string): number => BUDGET_MIN[id] ?? 1;
const budgetMax = (id: string): number => BUDGET_MAX[id] ?? 1;

/** What the trial has taught us about one provider so far. */
interface ProviderState {
  attempts: number;
  successes: number;
  consecutiveFailures: number;
}
const providerState = new Map<string, ProviderState>();
function stateOf(id: string): ProviderState {
  let s = providerState.get(id);
  if (s === undefined) {
    s = { attempts: 0, successes: 0, consecutiveFailures: 0 };
    providerState.set(id, s);
  }
  return s;
}
const isDead = (id: string): boolean => stateOf(id).consecutiveFailures >= 2;
const isDry = (id: string): boolean => stateOf(id).attempts >= budgetMax(id);

interface Candidate {
  provider: ProviderEntry;
  row: RatingRow;
}

/**
 * Every free model that meets the requirements, best rating first. On a
 * rating tie the provider whose wire matches the client's goes first (the
 * request then needs no translation at all), then the provider id, then the
 * model id, so the order is fixed.
 */
function candidatesFor(brief: Brief, requires: Requirement, clientWire: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const provider of brief.providers) {
    // An OpenAI client asking for JSON mode cannot be served by an Anthropic
    // provider: that wire carries no response_format, so the provider would
    // echo json_mode false and the request would fail its echo check. Gemini
    // carries it as responseMimeType, OpenAI natively.
    if (clientWire === 'openai' && requires.json === true && provider.wire === 'anthropic') continue;
    for (const row of brief.ratings) {
      if (row.provider !== provider.id || !row.free) continue;
      if (!meetsRequirements(requires, row)) continue;
      out.push({ provider, row });
    }
  }
  out.sort((a, b) => {
    if (a.row.rating_milli !== b.row.rating_milli) return b.row.rating_milli - a.row.rating_milli;
    const aSame = a.provider.wire === clientWire ? 0 : 1;
    const bSame = b.provider.wire === clientWire ? 0 : 1;
    if (aSame !== bSame) return aSame - bSame;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0;
  });
  return out;
}

/* ------------------------------------------------------------------ */
/* Wire helpers: reading request bodies and translating between wires */
/* ------------------------------------------------------------------ */

/** The text of a string or of a block array's text blocks; '' when none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {}. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason / Anthropic's stop_reason. */
function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}
function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** One Anthropic content block as an OpenAI content part, or null. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block, or null. */
function openAIPartToBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (dataUrl !== null) {
      return { type: 'image', source: { type: 'base64', media_type: dataUrl[1] ?? 'image/png', data: dataUrl[2] ?? '' } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: system messages become the system prompt, image_url parts become
 * image blocks, tool_calls become tool_use blocks, tool messages become
 * tool_result blocks, and the rest passes through. Anthropic's wire carries
 * no JSON mode: callers never route a JSON-mode request here.
 */
function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof content === 'string' ? content : toolResultText(content) }],
      });
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // User (and anything else): text and image parts become blocks.
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToBlock(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** OpenAI function tools as Gemini functionDeclarations. */
function toolsToGemini(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const declarations: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    declarations.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  return declarations;
}

/** Anthropic tools as Gemini functionDeclarations. */
function anthropicToolsToGemini(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const declarations: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    declarations.push({
      name: raw.name,
      description: typeof raw.description === 'string' ? raw.description : '',
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  return declarations;
}

/** One OpenAI message as Gemini contents; system messages go to the caller. */
function openAIMessageToGemini(raw: Json): Json | null {
  const role = raw.role;
  const content = raw.content;
  if (role === 'assistant') {
    const parts: Json[] = [];
    const text = textOf(content);
    if (text !== '') parts.push({ text });
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
    }
    return parts.length > 0 ? { role: 'model', parts } : null;
  }
  if (role === 'tool') {
    return {
      role: 'user',
      parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { result: typeof content === 'string' ? content : toolResultText(content) } } }],
    };
  }
  if (typeof content === 'string') return { role: 'user', parts: [{ text: content }] };
  if (!Array.isArray(content)) return null;
  const parts: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
      if (dataUrl !== null) parts.push({ inlineData: { mimeType: dataUrl[1] ?? 'image/png', data: dataUrl[2] ?? '' } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts.length > 0 ? { role: 'user', parts } : null;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
function openAIToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const content = openAIMessageToGemini(raw);
    if (content !== null) contents.push(content);
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const declarations = toolsToGemini(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** One Anthropic message as Gemini contents. */
function anthropicMessageToGemini(raw: Json): Json | null {
  const role = raw.role;
  const content = raw.content;
  if (role === 'assistant') {
    if (typeof content === 'string') return { role: 'model', parts: [{ text: content }] };
    if (!Array.isArray(content)) return null;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
      }
    }
    return parts.length > 0 ? { role: 'model', parts } : null;
  }
  if (typeof content === 'string') return { role: 'user', parts: [{ text: content }] };
  if (!Array.isArray(content)) return null;
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') parts.push({ fileData: { fileUri: source.url } });
    } else if (block.type === 'tool_result') {
      parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { result: toolResultText(block.content) } } });
    }
  }
  return parts.length > 0 ? { role: 'user', parts } : null;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = anthropicMessageToGemini(raw);
    if (content !== null) contents.push(content);
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = anthropicToolsToGemini(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** One SSE frame: `event: <name>` when named, then the JSON data line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split buffered SSE text into {event, data} frames. */
function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/* ------------------------------------------------------------------ */
/* Answers: provider responses back into the client's wire            */
/* ------------------------------------------------------------------ */

/** The text of Gemini candidates' parts, and any function calls in them. */
function geminiCandidate(answer: Json): { text: string; calls: Json[]; finish: unknown; usage: { input: number; output: number } } {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  const calls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    else if (isRecord(part.functionCall)) {
      calls.push({
        id: `call_${calls.length}`,
        type: 'function',
        function: { name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) },
      });
    }
  }
  const meta = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return {
    text: texts.join(''),
    calls,
    finish: first.finishReason,
    usage: { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) },
  };
}

/** A Chat Completions answer as an Anthropic Messages object. */
function openAIToAnthropicJson(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: openAIStopToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
function anthropicToOpenAIJson(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const texts: string[] = [];
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    else if (block.type === 'tool_use') {
      calls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: texts.join('') };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** A Gemini generateContent answer as a Chat Completions object. */
function geminiToOpenAIJson(body: Json, model: string): Json {
  const parsed = geminiCandidate(body);
  const message: Json = { role: 'assistant', content: parsed.text };
  if (parsed.calls.length > 0) message.tool_calls = parsed.calls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: geminiStopToOpenAI(parsed.finish), logprobs: null }],
    usage: { prompt_tokens: parsed.usage.input, completion_tokens: parsed.usage.output, total_tokens: parsed.usage.input + parsed.usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
function geminiToAnthropicJson(body: Json, model: string): Json {
  const parsed = geminiCandidate(body);
  const content: Json[] = [];
  if (parsed.text !== '') content.push({ type: 'text', text: parsed.text });
  for (const call of parsed.calls) {
    if (!isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: geminiStopToAnthropic(parsed.finish),
    stop_sequence: null,
    usage: { input_tokens: parsed.usage.input, output_tokens: parsed.usage.output },
  };
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start
 * on the first event, content_block_start/delta/stop per text block or tool
 * call, then message_delta with the stop reason and usage, then message_stop.
 * The whole upstream stream is already read, so message_start carries the
 * provider's real input tokens.
 */
function openAIStreamToAnthropic(text: string, model: string): string {
  let fullText = '';
  let finish: unknown = null;
  let inputTokens = 0;
  let outputTokens = 0;
  const calls: { id: string; name: string; args: string }[] = [];
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') fullText += delta.content;
    const toolCalls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of toolCalls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : calls.length;
      while (calls.length <= index) calls.push({ id: '', name: '', args: '' });
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof call.id === 'string' && call.id !== '') calls[index].id = call.id;
      if (typeof fn.name === 'string' && fn.name !== '') calls[index].name = fn.name;
      if (typeof fn.arguments === 'string') calls[index].args += fn.arguments;
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) finish = choice.finish_reason;
  }
  const out: string[] = [];
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
    }),
  );
  let nextIndex = 0;
  if (fullText !== '') {
    out.push(sseEvent('content_block_start', { type: 'content_block_start', index: nextIndex, content_block: { type: 'text', text: '' } }));
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: nextIndex, delta: { type: 'text_delta', text: fullText } }));
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: nextIndex }));
    nextIndex += 1;
  }
  for (const call of calls) {
    out.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: nextIndex,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${nextIndex}`, name: call.name, input: {} },
      }),
    );
    if (call.args !== '') {
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: nextIndex, delta: { type: 'input_json_delta', partial_json: call.args } }));
    }
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: nextIndex }));
    nextIndex += 1;
  }
  out.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: openAIStopToAnthropic(finish), stop_sequence: null }, usage: { output_tokens: outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return out.join('');
}

/**
 * An Anthropic SSE stream as OpenAI chat chunks: a role chunk, the text and
 * tool-call deltas, a finish chunk, a usage chunk with the provider's exact
 * counts, then [DONE].
 */
function anthropicStreamToOpenAI(text: string, model: string): string {
  let fullText = '';
  let stop: unknown = null;
  let inputTokens = 0;
  let outputTokens = 0;
  const calls: { id: string; name: string; args: string }[] = [];
  const byIndex = new Map<number, number>();
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block)) {
      const index = typeof data.index === 'number' ? data.index : 0;
      if (data.content_block.type === 'tool_use') {
        const at = calls.length;
        byIndex.set(index, at);
        calls.push({ id: typeof data.content_block.id === 'string' ? data.content_block.id : '', name: typeof data.content_block.name === 'string' ? data.content_block.name : '', args: '' });
      }
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const index = typeof data.index === 'number' ? data.index : 0;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') fullText += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const at = byIndex.get(index);
        if (at !== undefined) calls[at].args += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const out: string[] = [];
  out.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  if (fullText !== '') out.push(sseEvent(null, chunk({ content: fullText }, null)));
  calls.forEach((call, i) => {
    out.push(sseEvent(null, chunk({ tool_calls: [{ index: i, id: call.id !== '' ? call.id : `call_${i}`, type: 'function', function: { name: call.name, arguments: call.args } }] }, null)));
  });
  out.push(sseEvent(null, chunk({}, anthropicStopToOpenAI(stop))));
  out.push(
    sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } }),
  );
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A Gemini SSE stream as OpenAI chat chunks, with usage and [DONE]. */
function geminiStreamToOpenAI(text: string, model: string): string {
  let fullText = '';
  let finish: unknown = 'STOP';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string') fullText += part.text;
    }
    if (typeof first.finishReason === 'string') finish = first.finishReason;
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  const chunk = (delta: Json, fin: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: fin }],
  });
  const out: string[] = [];
  out.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  if (fullText !== '') out.push(sseEvent(null, chunk({ content: fullText }, null)));
  out.push(sseEvent(null, chunk({}, geminiStopToOpenAI(finish))));
  out.push(
    sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } }),
  );
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A Gemini SSE stream as Anthropic events, with usage and message_stop. */
function geminiStreamToAnthropic(text: string, model: string): string {
  let fullText = '';
  let finish: unknown = 'STOP';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string') fullText += part.text;
    }
    if (typeof first.finishReason === 'string') finish = first.finishReason;
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  const out: string[] = [];
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
    }),
  );
  if (fullText !== '') {
    out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: fullText } }));
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  }
  out.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: geminiStopToAnthropic(finish), stop_sequence: null }, usage: { output_tokens: outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return out.join('');
}

/* ------------------------------------------------------------------ */
/* Upstream calls and failover                                         */
/* ------------------------------------------------------------------ */

interface UpstreamResult {
  ok: boolean;
  status: number;
  code: string;
  text: string;
}

/** One call to one provider, in that provider's wire, body fully read. */
async function callProvider(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const model = candidate.row.model;
  let url: string;
  let upstreamBody: Json;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientWire === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    if (stream) {
      headers.accept = 'text/event-stream';
      upstreamBody = { ...upstreamBody, stream_options: { include_usage: true } };
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientWire === 'anthropic' ? { ...body, model } : openAIToAnthropic(body, model);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    upstreamBody = clientWire === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
    if (stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, code: 'unreachable', text: (err as Error).message };
  }
  const text = await res.text();
  let code = '';
  if (!res.ok) {
    try {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.code === 'string') code = parsed.error.code;
      else if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.type === 'string') code = parsed.error.type;
    } catch {
      // The error body is not JSON: keep the status only.
    }
  }
  return { ok: res.ok, status: res.status, code, text };
}

/** A provider's answer converted into the client's wire. */
function convertAnswer(
  candidate: Candidate,
  clientWire: 'openai' | 'anthropic',
  stream: boolean,
  text: string,
): { contentType: string; text: string } {
  const providerWire = candidate.provider.wire;
  const model = candidate.row.model;
  if (!stream) {
    if (providerWire === clientWire) return { contentType: 'application/json', text };
    let body: Json;
    try {
      const parsed: unknown = JSON.parse(text);
      body = isRecord(parsed) ? parsed : {};
    } catch {
      body = {};
    }
    if (clientWire === 'openai') {
      const converted = providerWire === 'anthropic' ? anthropicToOpenAIJson(body, model) : geminiToOpenAIJson(body, model);
      return { contentType: 'application/json', text: JSON.stringify(converted) };
    }
    const converted = providerWire === 'openai' ? openAIToAnthropicJson(body, model) : geminiToAnthropicJson(body, model);
    return { contentType: 'application/json', text: JSON.stringify(converted) };
  }
  if (providerWire === clientWire) return { contentType: 'text/event-stream', text };
  if (clientWire === 'openai') {
    const converted = providerWire === 'anthropic' ? anthropicStreamToOpenAI(text, model) : geminiStreamToOpenAI(text, model);
    return { contentType: 'text/event-stream', text: converted };
  }
  const converted = providerWire === 'openai' ? openAIStreamToAnthropic(text, model) : geminiStreamToAnthropic(text, model);
  return { contentType: 'text/event-stream', text: converted };
}

/** Requirements read off a request body, for the manifest's row to union with. */
function requirementsFromBody(clientWire: 'openai' | 'anthropic', body: Json): Requirement {
  let tools = false;
  let vision = false;
  let json = false;
  if (Array.isArray(body.tools) && body.tools.length > 0) tools = true;
  if (clientWire === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && (block.type === 'image_url' || block.type === 'image')) vision = true;
      }
    }
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    json = format === 'json_object' || format === 'json_schema';
  } else {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image') vision = true;
      }
    }
  }
  return { tools: tools ? true : undefined, vision: vision ? true : undefined, json: json ? true : undefined };
}

/* ------------------------------------------------------------------ */
/* The server                                                          */
/* ------------------------------------------------------------------ */

const brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The free models the catalog registers, each id once, provider order. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  const seen = new Set<string>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  return out;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
    });
  }
}

/**
 * Serve one client request: try the ranked candidates in order, with the
 * retry rule, then a last-resort pass, and answer with the first provider
 * that serves it. Returns null when no candidate served it.
 */
async function serve(
  clientWire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<{ contentType: string; text: string } | null> {
  const manifest = brief.workload[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(clientWire, body);
  const requires: Requirement = {
    tools: manifest?.requires.tools === true || fromBody.tools === true ? true : undefined,
    vision: manifest?.requires.vision === true || fromBody.vision === true ? true : undefined,
    json: manifest?.requires.json === true || fromBody.json === true ? true : undefined,
    ...(manifest?.requires.min_context !== undefined ? { min_context: manifest.requires.min_context } : {}),
  };
  const candidates = candidatesFor(brief, requires, clientWire);

  const attempt = async (candidate: Candidate): Promise<{ contentType: string; text: string } | null> => {
    const s = stateOf(candidate.provider.id);
    s.attempts += 1;
    const result = await callProvider(candidate, clientWire, body, stream);
    if (result.ok) {
      s.successes += 1;
      s.consecutiveFailures = 0;
      return convertAnswer(candidate, clientWire, stream, result.text);
    }
    s.consecutiveFailures += 1;
    log(`${candidate.provider.id}/${candidate.row.model} refused HTTP ${result.status}${result.code !== '' ? ` (${result.code})` : ''}`);
    return null;
  };

  // Pass 1: ranked candidates, skipping dead or certainly-dry providers, with
  // one immediate retry of a first consecutive refusal below the budget
  // minimum (a scheduled failure, not a spent budget). The retry costs the
  // same one call a failover would, and keeps the request on the best model.
  for (const candidate of candidates) {
    const id = candidate.provider.id;
    if (isDead(id) || isDry(id)) continue;
    const answered = await attempt(candidate);
    if (answered !== null) return answered;
    const s = stateOf(id);
    if (s.consecutiveFailures === 1 && s.attempts < budgetMin(id)) {
      const retried = await attempt(candidate);
      if (retried !== null) return retried;
    }
  }
  // Pass 2: nothing served yet; try every candidate once more, dead ones
  // included, in case two scheduled failures fell in a row.
  for (const candidate of candidates) {
    if (isDry(candidate.provider.id)) continue;
    const answered = await attempt(candidate);
    if (answered !== null) return answered;
  }
  return null;
}

async function handlePost(req: IncomingMessage, res: ServerResponse, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const answered = await serve(wire, body, stream);
  if (answered === null) {
    return sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const text = answered.text;
  const headers: Record<string, string> = { 'content-type': answered.contentType, 'content-length': Buffer.byteLength(text) };
  if (stream) {
    headers['cache-control'] = 'no-cache';
    headers.connection = 'keep-alive';
  }
  res.writeHead(200, headers);
  res.end(text);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handlePost(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handlePost(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number): Promise<void> {
  let lastError = '';
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = (err as Error).message;
      await sleep(200);
    }
  }
  throw new Error(`register failed: ${lastError}`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
