const ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions';

// Wide bursts draw 429s, so back off and retry rather than losing the item. Callers still
// persist per-item, so anything that exhausts its attempts is picked up by the next run.
// Read lazily: this module is evaluated before the entry point loads .env.
const attempts = () => Number(process.env.OPENROUTER_ATTEMPTS ?? 5);
// A ceiling on a single call, not a target: the slowest legitimate reply here is a
// reasoning-enabled batch of twenty scenes. It exists because a connection the provider
// accepts and never answers would otherwise hang its worker forever, and a stage running
// hundreds wide loses every worker the same way — going silent instead of failing.
const timeout = () => Number(process.env.OPENROUTER_TIMEOUT_S ?? 300) * 1000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export async function chat(body) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error('OPENROUTER_API_KEY is not set');
  const maxAttempts = attempts();

  for (let attempt = 1; ; attempt++) {
    let response;
    try {
      response = await fetch(ENDPOINT, {
        method: 'POST',
        headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeout()),
      });
    } catch (err) {
      // A timeout or a dropped socket is the same kind of transient as a 5xx, so it backs
      // off and re-rolls rather than losing the item.
      if (attempt >= maxAttempts) throw new Error(`OpenRouter unreachable: ${err.message}`);
      await sleep(2 ** attempt * 1000 * (0.5 + Math.random()));
      continue;
    }
    if (response.ok) return response.json();

    const detail = (await response.text()).slice(0, 200);
    const retryable = response.status === 429 || response.status >= 500;
    if (!retryable || attempt >= maxAttempts) throw new Error(`OpenRouter ${response.status}: ${detail}`);

    // Honour Retry-After when offered, otherwise exponential backoff with jitter so 50
    // workers do not all come back at the same instant.
    const retryAfter = Number(response.headers.get('retry-after'));
    const backoff = retryAfter > 0 ? retryAfter * 1000 : 2 ** attempt * 1000 * (0.5 + Math.random());
    await sleep(backoff);
  }
}

/**
 * The user message, as plain text or as text followed by images.
 *
 * Images go inline as base64 data URLs rather than links: the frames being asked about are
 * rendered on the fly and have no address anything else could fetch them from. A `Buffer` is
 * assumed to be a PNG; a string is passed through, so an already-formed data URL or a real
 * URL both work.
 */
const userContent = (text, images) => {
  if (!images?.length) return text;
  return [
    { type: 'text', text },
    ...images.map((image) => ({
      type: 'image_url',
      image_url: {
        url: Buffer.isBuffer(image) ? `data:image/png;base64,${image.toString('base64')}` : image,
      },
    })),
  ];
};

/** Structured-output call: returns the parsed JSON payload plus what it cost. `reasoning`
 *  is OpenRouter's unified thinking knob and is omitted entirely when not asked for. */
export async function chatJSON({ model, system, user, images, name, schema, temperature, reasoning }) {
  const body = await chat({
    model,
    ...(temperature === undefined ? {} : { temperature }),
    ...(reasoning ? { reasoning } : {}),
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: userContent(user, images) },
    ],
    response_format: { type: 'json_schema', json_schema: { name, strict: true, schema } },
  });

  const content = body.choices?.[0]?.message?.content;
  if (!content) throw new Error(`OpenRouter returned no content: ${JSON.stringify(body).slice(0, 500)}`);
  return { data: JSON.parse(content), model: body.model ?? model, usage: body.usage ?? null };
}
