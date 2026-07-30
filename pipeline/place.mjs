// Step 5: ask an LLM where B goes, given both objects as solid blocks and the placement
// phrase. Each object arrives as its voxel occupancy decomposed into maximal axis-aligned
// blocks in its own bbox-centred frame; the answer is expressed in A's frame.

import { chatJSON } from './openrouter.mjs';

const DEFAULT_MODEL = 'openai/gpt-5.6-luna';

export const SYSTEM = `You are a 3D scene composer. You get the solid geometry of an anchor object A and a movable object B, each as a list of axis-aligned solid blocks, plus a phrase describing how B should be placed relative to A. Return the transform that places B.

Coordinates are right-handed with +Y up and +Z as an object's front face (glTF convention).

Reading an object: each line is one solid block, \`(x0, y0, z0) to (x1, y1, z1)\`, in that object's own frame with its bounding-box centre at the origin. Blocks are disjoint, listed largest first, and together they make up the object. Space they do not cover is empty — gaps, recesses and cavities are real, and are where something can fit into or under the object.

The two objects are in unrelated units: A's numbers and B's numbers do not compare, and B's absolute size is undefined — you decide it. \`scale\` is the factor B's coordinates are multiplied by to express B in A's units, so B's bounding box ends up its printed size times \`scale\`.

Answer in A's frame:
- reasoning: pick B's real-world size relative to A and convert it to \`scale\`, read the target surface or cavity off A's blocks, then derive the centre. Do the arithmetic here.
- scale
- yaw_degrees: rotation about +Y by the right-hand rule — B's +Z front swings toward its +X side.
- pitch_degrees: rotation about B's +X by the right-hand rule — B's top tips toward its front.
- roll_degrees: rotation about B's +Z by the right-hand rule — B's +X side rises.
- center: where B's bounding-box centre lands, in A's frame.

Rotations apply in that order (yaw, then pitch, then roll), each about B's own axes as rotated so far, and all about B's bounding-box centre — so rotation never moves the centre. Most placements sit upright and want pitch and roll of 0; use them when the phrase genuinely tilts B, like leaning or propping it against something.

Contact must be physically plausible: something resting on a surface touches it with no gap and no interpenetration, something inside a cavity fits within its walls and rests on its floor, something mounted touches the face it attaches to, and something leaning touches both the ground and the face it leans on.

After you answer, a physics pass refines B against the true meshes — the blocks are coarse, so B may land a sliver off the real surfaces. Classify the contact the phrase intends so the pass knows what to do:
- contact: "rest" when gravity holds B against A or the ground (on top of, on a shelf, inside an open container, on the floor beside); the pass drops B onto the true surface and lets it settle. "lean" when gravity presses B against both the ground and a face of A; the pass seats both contacts. "attach" when B is held touching A without gravity's help (mounted on a face, hanging from above, stuck to a side); the pass slides B along contact_direction into exact contact. "embed" when B intentionally penetrates A (a dart in a dartboard, an axe in a stump); the pass drives B along contact_direction until embed_fraction of B is buried. "drape" when B is soft — a blanket, a towel, a cape, a folded jacket — and should deform over A under gravity; the pass simulates it as cloth and bakes the draped shape. "none" when no contact is intended (floating, hovering); the pass leaves B exactly where you put it.
- contact_direction: for lean, attach and embed, the direction B moves or presses to meet A, as a vector in A's frame; [0, 0, 0] lets the pass use the shortest way from B to A. Use [0, 0, 0] for rest and none.
- embed_fraction: for embed, the fraction of B's extent along contact_direction that ends up inside A (0.05 is a shallow tip, 0.5 half-buried). 0 for every other contact.`;

export const CONTACTS = ['rest', 'lean', 'attach', 'embed', 'drape', 'none'];

const SCHEMA = {
  type: 'object',
  properties: {
    reasoning: { type: 'string' },
    scale: { type: 'number' },
    yaw_degrees: { type: 'number' },
    pitch_degrees: { type: 'number' },
    roll_degrees: { type: 'number' },
    center: { type: 'array', items: { type: 'number' } },
    contact: { type: 'string', enum: CONTACTS },
    contact_direction: { type: 'array', items: { type: 'number' } },
    embed_fraction: { type: 'number' },
  },
  required: [
    'reasoning', 'scale', 'yaw_degrees', 'pitch_degrees', 'roll_degrees', 'center',
    'contact', 'contact_direction', 'embed_fraction',
  ],
  additionalProperties: false,
};

function describe(heading, { name, grid, blocks }) {
  const size = grid.size.map((v) => Number(v.toFixed(4)));
  return [
    `${heading} — "${name}"`,
    `bounding box ${size.join(' x ')}  (x, y, z), centred at the origin — ${blocks.length} block(s), largest first`,
    ...blocks.map(([x0, y0, z0, x1, y1, z1]) => `(${x0}, ${y0}, ${z0}) to (${x1}, ${y1}, ${z1})`),
  ].join('\n');
}

export function buildPrompt({ phrase, a, b }) {
  return [
    `PLACEMENT: "${phrase}"`,
    '',
    describe('A (anchor, stays fixed)', a),
    '',
    describe('B (place this)', b),
  ].join('\n');
}

/**
 * OpenRouter's unified reasoning knob, from `OPENROUTER_REASONING`: thinking is disabled
 * unless an effort level is asked for — placement answers proved as good without it, and
 * reasoning tokens were most of every call's cost. `default` restores the provider's own
 * behaviour, anything else is an effort level.
 */
function reasoningConfig() {
  const value = (process.env.OPENROUTER_REASONING ?? 'off').trim().toLowerCase();
  if (value === 'default') return undefined;
  return value === 'off' ? { enabled: false } : { effort: value };
}

export async function requestPlacement({ phrase, a, b, model = process.env.OPENROUTER_MODEL ?? DEFAULT_MODEL }) {
  const prompt = buildPrompt({ phrase, a, b });
  const { data, model: used, usage } = await chatJSON({
    model,
    system: SYSTEM,
    user: prompt,
    name: 'placement',
    schema: SCHEMA,
    reasoning: reasoningConfig(),
  });

  const valid =
    Number.isFinite(data.scale) && data.scale > 0 &&
    [data.yaw_degrees, data.pitch_degrees, data.roll_degrees].every(Number.isFinite) &&
    Array.isArray(data.center) && data.center.length === 3 && data.center.every(Number.isFinite) &&
    CONTACTS.includes(data.contact) &&
    Array.isArray(data.contact_direction) && data.contact_direction.length === 3 &&
    data.contact_direction.every(Number.isFinite) &&
    Number.isFinite(data.embed_fraction);
  if (!valid) throw new Error(`invalid placement from model: ${JSON.stringify(data).slice(0, 500)}`);

  // The prompt rides along so callers can log exactly what the model saw, not a rebuild.
  return { ...data, prompt, model: used, usage };
}
