// Step 5: ask an LLM where B goes, given both voxel grids and the placement phrase.
// The answer is expressed entirely in "A-voxels" — A's grid is the working frame.

import { chatJSON } from './openrouter.mjs';

const DEFAULT_MODEL = 'google/gemini-3.1-pro-preview';

const SYSTEM = `You are a 3D scene composer. You get voxel occupancy grids for an anchor object A and a movable object B, plus a phrase describing how B should be placed relative to A. Return the transform that places B.

Coordinates are right-handed with +Y up and +Z as an object's front face (glTF convention).

Reading a grid: it is printed as horizontal slices, top layer first. Inside a slice each line is one Z index (first line z=0, last line z=dz-1) and each character is one X index starting at x=0; '#' is solid, '.' is empty. Cells are cubes, so a grid's dimensions are proportional to the object's real proportions. Consecutive layers that are identical are printed once under a range header like \`y=31..12\`, meaning every layer from 31 down to 12 looks like that.

Working units are A-voxels: A's grid is the frame, so cell (i,j,k) of A spans [i,i+1] x [j,j+1] x [k,k+1] and A's grid occupies [0,adx] x [0,ady] x [0,adz]. Coordinates are continuous, may be fractional, and may fall outside A's grid.

B's grid only tells you B's shape and proportions — its absolute size is arbitrary and you decide it. \`scale\` is how many A-voxels one B-voxel becomes, so B's bounding box ends up bdx*scale by bdy*scale by bdz*scale A-voxels.

Answer with:
- reasoning: pick B's real-world size relative to A and convert it to \`scale\`, read the target surface or cavity off A's slices, then derive the centre. Do the arithmetic here.
- scale
- yaw_degrees: B's rotation about +Y by the right-hand rule (B's +Z front swings toward +X).
- center: B's bounding-box centre in A-voxel coordinates. B rotates about that centre, so yaw never moves it.

Contact must be physically plausible: something resting on a surface touches it with no gap and no interpenetration, something inside a cavity fits within its walls and rests on its floor, something mounted touches the face it attaches to.`;

const SCHEMA = {
  type: 'object',
  properties: {
    reasoning: { type: 'string' },
    scale: { type: 'number' },
    yaw_degrees: { type: 'number' },
    center: { type: 'array', items: { type: 'number' } },
  },
  required: ['reasoning', 'scale', 'yaw_degrees', 'center'],
  additionalProperties: false,
};

export function describe(heading, { name, grid, slices }) {
  const [dx, dy, dz] = grid.dims;
  const lines = [`${heading} — "${name}"`, `grid ${dx} x ${dy} x ${dz}  (x, y, z)`];
  for (let y = dy - 1; y >= 0; ) {
    let lowest = y;
    while (lowest > 0 && slices[lowest - 1] === slices[y]) lowest--;
    const label = lowest === y ? `y=${y}` : `y=${y}..${lowest}`;
    lines.push(slices[y].includes('#') ? `--- ${label} ---\n${slices[y]}` : `--- ${label} --- (empty)`);
    y = lowest - 1;
  }
  return lines.join('\n');
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

export async function requestPlacement({ phrase, a, b, model = process.env.OPENROUTER_MODEL ?? DEFAULT_MODEL }) {
  const { data, model: used, usage } = await chatJSON({
    model,
    system: SYSTEM,
    user: buildPrompt({ phrase, a, b }),
    name: 'placement',
    schema: SCHEMA,
  });

  const valid =
    Number.isFinite(data.scale) && data.scale > 0 &&
    Number.isFinite(data.yaw_degrees) &&
    Array.isArray(data.center) && data.center.length === 3 && data.center.every(Number.isFinite);
  if (!valid) throw new Error(`invalid placement from model: ${JSON.stringify(data).slice(0, 500)}`);

  return { ...data, model: used, usage };
}
