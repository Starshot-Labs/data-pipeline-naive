// Nano Banana on Google's Generative Language API, the one stage that does not go through
// OpenRouter. The settings that decide whether an image is any use to stage 3 — square
// framing, 512 px, thinking level — only exist on the native endpoint.
//
// Everything leaves here as PNG, whatever the model chose to answer with.

import sharp from 'sharp';

const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models';

// Trellis reconstructs from one square reference photo, and 512 is both the cheapest and
// the quickest size the model offers. HIGH thinking is slower than MINIMAL but returns
// cleaner single objects.
const ASPECT_RATIO = '1:1';
const IMAGE_SIZE = '512';
const THINKING_LEVEL = 'HIGH';

// What the model may answer with. All of them are re-encoded to PNG on the way out.
const DECODABLE = new Set(['image/png', 'image/jpeg', 'image/webp']);

// An empty response the model clears on a re-roll. Every other finish reason — SAFETY,
// PROHIBITED_CONTENT — is a verdict on the prompt, and re-issuing it buys the same verdict.
const TRANSIENT_FINISH = new Set(['MALFORMED_FUNCTION_CALL', 'FINISH_REASON_UNSPECIFIED', 'OTHER']);

// Read lazily: this module is evaluated before the entry point loads .env.
const maxAttempts = () => Number(process.env.GOOGLE_ATTEMPTS ?? 5);

// Rate limits and recitation refusals get their own budgets, because neither is a verdict
// on the request: a 429 is fifty workers colliding, and the recitation filter fires
// probabilistically and almost always clears on the next roll. Charging either to the
// general budget would leave nothing for the failures that mean something.
const RATE_LIMIT_ATTEMPTS = 8;
const RECITATION_ATTEMPTS = 15;
const FALLBACK_RETRY_S = 30;
const RECITATION_BACKOFF_MS = 500;
const RECITATION_BACKOFF_CAP_MS = 5000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// One deadline every caller waits behind, so a 429 holds up the whole stage rather than
// only the worker that drew it — otherwise the other forty-nine keep hammering a quota
// that has already said no. Google's own retryDelay sets how long.
let openAt = 0;
const hold = (seconds) => {
  openAt = Math.max(openAt, Date.now() + seconds * 1000);
};
async function waitTurn() {
  for (let wait = openAt - Date.now(); wait > 0; wait = openAt - Date.now()) await sleep(wait);
}

/** Google's RetryInfo hint out of a 429 body, e.g. `"27s"` → 27. */
function retryDelay(detail) {
  const seconds = /"retryDelay"\s*:\s*"([\d.]+)s"/.exec(detail)?.[1];
  return seconds ? Number(seconds) : null;
}

/** One request. Either the image, or how the failure wants to be retried. */
async function attempt({ apiKey, model, prompt, images }) {
  // Input images go ahead of the text, so a prompt can point at them as
  // "image 1", "image 2" in the order the caller supplied them.
  const parts = [
    ...images.map(({ mimeType, data }) => ({ inlineData: { mimeType, data: data.toString('base64') } })),
    { text: prompt },
  ];

  let response;
  try {
    response = await fetch(`${BASE_URL}/${model}:generateContent`, {
      method: 'POST',
      headers: { 'x-goog-api-key': apiKey, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ role: 'user', parts }],
        generationConfig: {
          imageConfig: { aspectRatio: ASPECT_RATIO, imageSize: IMAGE_SIZE },
          thinkingConfig: { thinkingLevel: THINKING_LEVEL, includeThoughts: false },
        },
      }),
    });
  } catch (err) {
    return { retry: 'transient', reason: `Google unreachable: ${err.message}` };
  }

  if (!response.ok) {
    const detail = await response.text();
    const reason = `Google ${response.status}: ${detail.slice(0, 200)}`;
    if (response.status === 429) return { retry: 'rateLimit', seconds: retryDelay(detail) ?? FALLBACK_RETRY_S, reason };
    return { retry: response.status >= 500 ? 'transient' : null, reason };
  }

  const candidate = (await response.json()).candidates?.[0];
  const inline = candidate?.content?.parts?.find((part) => part.inlineData?.data)?.inlineData;
  if (inline) {
    if (!DECODABLE.has(inline.mimeType)) {
      return { retry: null, reason: `Google returned ${inline.mimeType}, which nothing downstream reads` };
    }
    const bytes = Buffer.from(inline.data, 'base64');
    // The model picks a format per image — the last corpus came back roughly half JPEG —
    // and a dataset whose extensions have to be discovered before a file can be opened is a
    // tax on every consumer of it. Re-encoding adds no loss on top of what the model did.
    return { image: inline.mimeType === 'image/png' ? bytes : await sharp(bytes).png().toBuffer() };
  }

  const finish = candidate?.finishReason ?? 'no candidate';
  if (finish === 'IMAGE_RECITATION') return { retry: 'recitation', reason: 'recitation refusal' };
  return { retry: TRANSIENT_FINISH.has(finish) ? 'transient' : null, reason: `no image returned (finishReason=${finish})` };
}

/**
 * Text-to-image, returning PNG bytes and retrying on its own rather than handing the failure
 * back. Callers persist per image and the next run re-renders whatever is missing, so giving
 * up early is only ever paid for later.
 *
 * `images` optionally conditions the generation on reference images, each
 * `{ mimeType, data }` with `data` as raw bytes.
 */
export async function generateImage({ prompt, model, images = [] }) {
  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) throw new Error('GOOGLE_API_KEY is not set');
  const budget = maxAttempts();

  let transient = 0;
  let limited = 0;
  let recited = 0;

  for (;;) {
    await waitTurn();
    const { image, retry, seconds, reason } = await attempt({ apiKey, model, prompt, images });
    if (image) return image;

    if (retry === 'rateLimit') {
      hold(seconds);
      if (++limited >= RATE_LIMIT_ATTEMPTS) throw new Error(reason);
    } else if (retry === 'recitation') {
      if (++recited >= RECITATION_ATTEMPTS) throw new Error(reason);
      await sleep(Math.min(RECITATION_BACKOFF_CAP_MS, RECITATION_BACKOFF_MS * 2 ** recited));
    } else if (retry === 'transient' && ++transient < budget) {
      // Jittered so fifty workers do not all come back at the same instant.
      await sleep(2 ** transient * 1000 * (0.5 + Math.random()));
    } else {
      throw new Error(reason);
    }
  }
}
