// Step 2: turn an object description into a reference photo for image-to-3D.
// View angles mirror the manual exporter: the anchor is shot dead-on from the
// front with a 10° downward pitch, the placed object from an arbitrary direction.

import { generateImage } from './nano-banana.mjs';

const DEFAULT_MODEL = 'gemini-3.1-flash-image-preview';

export const ANCHOR_VIEW = { azimuth_deg: 0, elevation_deg: 10 };

export const randomView = () => ({
  azimuth_deg: Math.round((Math.random() * 2 - 1) * 160),
  elevation_deg: Math.round(5 + Math.random() * 30),
});

export function imagePrompt(description, view) {
  const side = view.azimuth_deg > 0 ? 'right' : 'left';
  const azimuth = view.azimuth_deg === 0
    ? 'straight-on from the front'
    : `from ${Math.abs(view.azimuth_deg)} degrees to the ${side} of straight-on`;
  return [
    description,
    `Photographed ${azimuth}, with the camera raised ${view.elevation_deg} degrees above the object so its top surface is slightly visible.`,
    'Product photograph of this single object alone on a plain flat white background,',
    'soft even studio lighting, no cast shadows on the backdrop, nothing else in shot,',
    'no text or watermarks, the whole object inside the frame with a small margin.',
    'Square 1:1 framing.',
  ].join(' ');
}

/** PNG bytes for one object, whatever format the model chose to answer with. */
export const renderImage = ({ prompt, model = process.env.IMAGE_MODEL ?? DEFAULT_MODEL }) =>
  generateImage({ prompt, model });
