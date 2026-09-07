// Progress reporting for stages that work through thousands of items.
//
// Two audiences, one API. On a terminal this rewrites a single line with a bar; in a
// captured log — a Modal container, a CI job, a file — carriage returns produce garbage, so
// it emits a whole line on an interval instead. Either way a stage that is going to take
// twenty minutes says so within the first second rather than looking hung, which is the
// failure this exists to prevent: a silent synchronous walk over a network volume is
// indistinguishable from a crash.
//
// Rate and ETA come from the overall average rather than a sliding window. Volume reads are
// bursty enough that an instantaneous rate swings wildly, and the number people act on is
// "when will this finish", which the average answers more honestly.

const BAR = 24;
const isTTY = () => Boolean(process.stdout.isTTY);

export function duration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return '—';
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

export const rateOf = (n, ms) => (ms > 0 ? (n / ms) * 1000 : 0);

const number = (n) => n.toLocaleString('en-US');

/**
 * A progress tracker over `total` items.
 *
 * `tick(n)` advances it, `note()` attaches a trailing string to the next render (a running
 * cost, an error count), and `done()` prints the final summary line. Renders are throttled
 * to `interval` so a tight loop cannot spend its time formatting.
 */
export function track(label, total, { interval = 2000, indent = '  ', heartbeat = 30000 } = {}) {
  const started = Date.now();
  const tty = isTTY();
  let count = 0;
  let last = 0;
  let extra = '';
  let dirty = false;

  const render = (final = false) => {
    const elapsed = Date.now() - started;
    const rate = rateOf(count, elapsed);
    const parts = [`${number(count)}${total ? `/${number(total)}` : ''}`];
    if (total) parts.push(`${((count / total) * 100).toFixed(0)}%`);
    parts.push(`${rate >= 100 ? rate.toFixed(0) : rate.toFixed(1)}/s`);
    parts.push(final ? duration(elapsed) : `eta ${total && rate > 0 ? duration(((total - count) / rate) * 1000) : '—'}`);
    if (extra) parts.push(extra);

    const body = `${label} ${parts.join(' · ')}`;
    if (tty && !final) {
      const filled = total ? Math.round((count / total) * BAR) : 0;
      const bar = total ? `[${'='.repeat(filled)}${' '.repeat(BAR - filled)}] ` : '';
      process.stdout.write(`\r${indent}${bar}${body}`.padEnd(90).slice(0, 200));
    } else {
      if (tty) process.stdout.write('\r');
      console.log(`${indent}${final ? '✓ ' : '… '}${body}`);
    }
    dirty = !final && tty;
    last = Date.now();
  };

  // A stage whose last few items are slow — one oversized mesh, one cloth solve — stops
  // ticking and would otherwise go silent for minutes, which is the state that reads as a
  // hang. The heartbeat keeps saying the elapsed time even when the count does not move.
  // Unref'd so it never keeps the process alive on its own.
  const beat = setInterval(() => {
    if (Date.now() - last >= heartbeat) render();
  }, Math.min(heartbeat, 10000));
  beat.unref?.();

  return {
    tick(n = 1) {
      count += n;
      if (Date.now() - last >= interval || count === total) render();
    },
    note(text) {
      extra = text;
    },
    /** Clears a half-drawn TTY line so an unrelated log does not land on top of it. */
    clear() {
      if (dirty) {
        process.stdout.write(`\r${' '.repeat(90)}\r`);
        dirty = false;
      }
    },
    done(text) {
      clearInterval(beat);
      if (text) extra = text;
      render(true);
      return { count, elapsed: Date.now() - started };
    },
    get count() {
      return count;
    },
  };
}

/** One-shot timing line for a step that has no countable items. */
export async function timed(label, work, { indent = '  ' } = {}) {
  const started = Date.now();
  const result = await work();
  console.log(`${indent}✓ ${label} · ${duration(Date.now() - started)}`);
  return result;
}
