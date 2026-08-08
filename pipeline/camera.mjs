// Where to look at the anchor from.
//
// VoxHammer renders five views and leaves the choice of one to the eye. All five sit at
// elevation 15, an almost level orbit, which is fine for a region on top of a table and
// useless for one inside a mug: measured on a real sample, elevation 15 and 30 saw none of
// the marked region at all, 60 saw 2.8% of the frame and 75 saw 3.7%. Sweeping azimuth
// cannot fix an enclosure that opens upward, so the angle has to be chosen rather than
// searched, and choosing it means reading the anchor's shape — which is an LLM's job.
//
// The angles are in the same frame the mask box is authored in: +Y up, +Z the anchor's
// front, so the model never leaves the coordinates it just read the occupancy grid in.
// modal/voxhammer_render.py is the one place that converts to Blender's.
//
// Three candidates rather than one because the model is reading slices, not renders: it can
// tell that a cavity opens upward, but not that a handle or a chair back clips the region at
// one particular azimuth. The service renders all three and keeps the best-ranked one whose
// mask actually survived, so the ranking is the preference and the depth test is the veto.

import { describe } from './place.mjs';
import { chatJSON } from './openrouter.mjs';

/** More than this and the render is paying for angles the ranking will never reach. */
const MAX_CAMERAS = 3;

// The poles leave the camera's up vector undefined, and looking straight down a cavity is
// no better than looking down it at 85.
const ELEVATION_LIMIT = 85;

const SYSTEM = `You are choosing where to put the camera for a 3D editing job. An anchor object A has already been generated, an object B is about to be placed relative to it, and the region B may occupy has already been marked as a box of grid cells.

Exactly one view of A is rendered from the angle you choose. B is painted into that view, and the editor then compares the painted view against the clean one to work out what changed. So a bad angle is fatal rather than merely untidy: if A's own geometry hides the marked region there is nothing on screen to paint into, and the job fails outright.

The grid is the generator's own lattice: exactly 64 x 64 x 64 cells over the cube A was normalized into, so A's longest axis spans all 64 cells and its shorter axes leave empty cells on either side of it. Indices are integers 0..63 on every axis — i along +X, j along +Y (up), k along +Z (A's front face), right-handed.

Reading a grid: it is printed as horizontal slices, top layer first. Inside a slice each line is one Z index (first line z=0, last line z=dz-1) and each character is one X index starting at x=0; '#' is solid, '.' is empty. Cells are cubes, so the grid's proportions are A's real proportions. Consecutive layers that are identical are printed once under a range header like \`y=31..12\`, meaning every layer from 31 down to 12 looks like that.

The camera always points at the centre of the cube from outside it, and you place it with two angles:
- elevation: degrees above the horizontal. 0 is level with the centre and looking horizontally, 85 is nearly overhead and looking straight down, negative looks up from below. Use -30..85.
- azimuth: degrees around the vertical axis, which is to say the side of A the camera stands on. 0 puts it on the +Z side, looking at A's front face; 90 on the +X side; 180 behind, looking at the back; 270 on the -X side. Use 0..359.

What decides the answer is how the marked region is enclosed:
- Open to one side — on a table top, on a seat, against a wall. A low elevation of 10 to 30 reads most naturally. Aim the azimuth at the side of A the region sits on: a region at high i wants azimuth 90, at low i azimuth 270, at high k azimuth 0, at low k azimuth 180, and a region in a corner wants an angle between the two.
- Recessed behind an opening — inside a mug, a bowl, a drawer, an open box. The camera has to look in through the opening or the wall hides the region completely. For a cavity that opens upward that means an elevation of 60 to 85, and this is exactly the case a level view fails on.
- Under an overhang — beneath a table, inside a shelf. A low or slightly negative elevation, looking in from whichever side is open.

Work out which of the region's six sides A's cells wall off and which are clear, then look in through a clear one.

Answer with:
- reasoning: say where the region sits inside A, which of its sides A's cells block and which are open, and therefore the direction the camera has to look from. Read the slices and do this before choosing any numbers.
- cameras: three candidates, best first. The first is the one you actually want. The other two are tried only if A's geometry turns out to hide more than the slices showed, so make them genuinely different — a second and third within a few degrees of the first are wasted.`;

const SCHEMA = {
  type: 'object',
  properties: {
    reasoning: { type: 'string' },
    cameras: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          elevation: { type: 'number' },
          azimuth: { type: 'number' },
        },
        required: ['elevation', 'azimuth'],
        additionalProperties: false,
      },
    },
  },
  required: ['reasoning', 'cameras'],
  additionalProperties: false,
};

export const buildCameraPrompt = ({ phrase, object, box, anchor }) =>
  [
    `PLACEMENT: "${phrase}"`,
    `OBJECT TO PLACE: ${object}`,
    `REGION ALREADY MARKED: cells [${box.min}] to [${box.max}], inclusive`,
    '',
    describe('A (anchor, already generated)', anchor),
  ].join('\n');

const clampElevation = (value) => Math.max(-ELEVATION_LIMIT, Math.min(ELEVATION_LIMIT, value));
const wrapAzimuth = (value) => ((value % 360) + 360) % 360;

/** The model's ranked angles, straightened onto the ranges the renderer accepts. */
export async function requestCamera({ phrase, object, box, anchor, model }) {
  const { data, model: used, usage } = await chatJSON({
    model,
    system: SYSTEM,
    user: buildCameraPrompt({ phrase, object, box, anchor }),
    name: 'camera_angles',
    schema: SCHEMA,
  });

  // An angle just past a limit is worth straightening rather than re-rolling, the same way
  // the mask box is. A reply with no usable angle at all is not.
  const cameras = (data.cameras ?? [])
    .filter((camera) => Number.isFinite(camera?.elevation) && Number.isFinite(camera?.azimuth))
    .slice(0, MAX_CAMERAS)
    .map((camera) => ({
      elevation: Math.round(clampElevation(camera.elevation)),
      azimuth: Math.round(wrapAzimuth(camera.azimuth)),
    }));

  if (!cameras.length) throw new Error(`no usable camera from model: ${JSON.stringify(data).slice(0, 500)}`);
  return { reasoning: data.reasoning, cameras, model: used, usage };
}
