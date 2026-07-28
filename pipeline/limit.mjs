// How wide a stage runs, and how many goes each item gets.

// Read lazily: this module is evaluated before an entry point loads .env.
const attempts = () => Number(process.env.MODEL_ATTEMPTS ?? 3);

/** Run `worker` over `items` with at most `limit` in flight. Results keep input order. */
export async function mapLimit(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (next < items.length) {
        const index = next++;
        results[index] = await worker(items[index]);
      }
    }),
  );
  return results;
}

/**
 * How many of a stage's calls run at once. Each stage has its own knob because the quotas
 * they answer to are unrelated — an image quota tightening is no reason to place fewer
 * samples — with `CONCURRENCY` left as a single lever to throttle everything at once.
 */
export const widthOf = (name, fallback = 200) => Number(process.env[name] ?? process.env.CONCURRENCY ?? fallback);

/**
 * Runs `work` until it resolves, `MODEL_ATTEMPTS` times at most.
 *
 * The API clients already retry what their APIs say about themselves: a 429's `retryDelay`,
 * a recitation refusal, a 5xx. This is the outer layer for everything they call final — a
 * model that answers with no image at all, a reply that will not parse, an answer that fails
 * validation. Those are usually a bad roll rather than a bad request, and re-rolling costs
 * one call against the entire extra pipeline pass it otherwise takes to notice.
 */
export async function retry(work) {
  const budget = attempts();
  for (let attempt = 1; ; attempt++) {
    try {
      return await work(attempt);
    } catch (err) {
      if (attempt >= budget) throw err;
      // Jittered so a whole stage's worth of failures does not come back in lockstep.
      await new Promise((resolve) => setTimeout(resolve, 2 ** attempt * 500 * (0.5 + Math.random())));
    }
  }
}
