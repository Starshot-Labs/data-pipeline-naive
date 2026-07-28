// Step 1: invent dataset samples — an anchor object, an object placed against it,
// visual descriptions for both, and the placement phrase.

import { chatJSON } from "./openrouter.mjs";

const DEFAULT_MODEL = "google/gemini-3.1-pro-preview";

export const CONTEXTS = [
	"mario platformer level",
	"walk-in closet",
	"candy kingdom",
	"retro arcade",
	"mudroom",
	"minecraft",
	"advanced alien civilization",
	"wizard alchemy tower",
	"laboratory",
	"workshop",
	"pirate ship deck",
	"kids playroom",
	"space station module",
	"kitchen",
	"haunted mansion",
	"campsite",
	"bathroom",
	"modular cardboard robot body",
];

const SYSTEM = `You invent samples for a 3D scene-composition dataset. Each sample is one large anchor object, one smaller object placed against it, and a short phrase describing that placement.

For every sample give:
- anchor_name / object_name: the object in one to three lowercase words ("dining table", "mario 1up mushroom", "tree stump", "modular robot torso").
- anchor_description / object_description: one sentence describing what the object looks like by itself — form, material, colour, proportions, style. This is fed straight to an image generator, so describe the object only: never a room, a background, a surface it stands on, another object, lighting or a camera.
- placement: a short natural phrase for where the object goes — the spatial relation, plus an optional detail about where on the anchor it sits. Which of the two object names it is allowed to mention changes per batch and is stated in the request; follow that exactly. try to use the most natural sentence structure, don't strictly follow any particular structure (as long as you're following the anchor/object placement rules) ex. X leaning against the right side of the front face, on top of Y, on the back right side, attached to the far corner of Y, inside the center of X 

Rules:
- the geometry of the anchor object should not be overly complex and is easily defined
- The anchor is the larger, load-bearing object; the other must plausibly: on, inside, hang from, underneath, lean against, etc. we should use simple language, nothing complex requiring potential transforms to the geometry (BAD: plugged into, draped over, etc.)
- DO NOT specify attributes of the anchor object that are non-relative, GOOD: on the top of the grandfather clock, on the front of the spaceship on the top left side BAD: on the arcade machine next to the joystick, attached to the side of the house next to the window
- Both must be single rigid objects that an image-to-3D model can reconstruct from one photo. No liquids, no cloth draped over things, no crowds of small parts.
- try to vary the spatial relation across the batch as much as possible, don't repeat the same type of relationship too much: on top of, inside, underneath, leaning against, mounted on, hanging from, tucked beside, straddling, etc.
- Every pair in the batch must be distinct, and each sample must be plausible in a real space.`;

const SAMPLE = {
	type: "object",
	properties: {
		anchor_name: { type: "string" },
		anchor_description: { type: "string" },
		object_name: { type: "string" },
		object_description: { type: "string" },
		placement: { type: "string" },
	},
	required: [
		"anchor_name",
		"anchor_description",
		"object_name",
		"object_description",
		"placement",
	],
	additionalProperties: false,
};

const SCHEMA = {
	type: "object",
	properties: { samples: { type: "array", items: SAMPLE } },
	required: ["samples"],
	additionalProperties: false,
};

/**
 * Which of the two object names the placement phrase may mention. One is drawn per request
 * so the dataset covers instructions that name both objects, either one, or neither — a
 * real instruction is as likely to be "on top" as "mug on top of the kitchen island".
 */
const PHRASINGS = {
	both: {
		form: "<object> <placement> <anchor>",
		rule: "name both the placed object and the anchor",
		example:
			"espresso machine on top of the kitchen island, near the back left corner",
	},
	anchor: {
		form: "<placement> <anchor>",
		rule: "name the anchor, and never the placed object",
		example: "on top of the kitchen island, near the back left corner",
	},
	object: {
		form: "<object> <placement>",
		rule: "name the placed object, and never the anchor",
		example:
			"espresso machine on the top surface, near the back left corner",
	},
	bare: {
		form: "<placement>",
		rule: 'name neither object"',
		example: "on the top surface, near the back left corner",
	},
};

export const PHRASINGS_KEYS = Object.keys(PHRASINGS);

const randomPhrasing = () =>
	PHRASINGS_KEYS[Math.floor(Math.random() * PHRASINGS_KEYS.length)];

/**
 * Spread `total` samples evenly over every context, split into requests of at most `batch`.
 *
 * Contexts are shuffled so runs too small to reach all of them do not keep picking the same
 * few. Phrasings are then dealt round-robin across the requests rather than drawn per
 * request: drawing independently leaves the mix to luck, and a ten-sample run could come out
 * eight `anchor` and one of everything else. The starting form is offset randomly so tiny
 * runs are not always `both` first.
 */
export function planRequests(total, batch) {
	const shuffled = CONTEXTS.map((context) => ({
		context,
		order: Math.random(),
	}))
		.sort((a, b) => a.order - b.order)
		.map((entry) => entry.context);

	const each = Math.floor(total / shuffled.length);
	const remainder = total % shuffled.length;
	const offset = Math.floor(Math.random() * PHRASINGS_KEYS.length);

	const plan = [];
	for (const [i, context] of shuffled.entries()) {
		for (
			let left = each + (i < remainder ? 1 : 0);
			left > 0;
			left -= batch
		) {
			plan.push({
				context,
				count: Math.min(left, batch),
				phrasing:
					PHRASINGS_KEYS[
						(plan.length + offset) % PHRASINGS_KEYS.length
					],
			});
		}
	}
	return plan;
}

export async function proposeSamples({
	count,
	context,
	phrasing = randomPhrasing(),
	avoid = [],
	model = process.env.SPEC_MODEL ?? DEFAULT_MODEL,
}) {
	const shape = PHRASINGS[phrasing];
	if (!shape)
		throw new Error(
			`unknown phrasing "${phrasing}" — expected one of ${PHRASINGS_KEYS.join(", ")}`,
		);

	const user = [
		`Generate ${count} samples set in a ${context}.`,
		`\nPlacement wording for this batch: write every placement as \`${shape.form}\`, so it must ${shape.rule}.`,
		` An espresso machine going onto a kitchen island would read "${shape.example}".`,
		avoid.length
			? `\n\nThese anchors are already covered — pick different ones:\n${avoid.join(", ")}`
			: "",
	].join("");

	const { data } = await chatJSON({
		model,
		system: SYSTEM,
		user,
		name: "samples",
		schema: SCHEMA,
		temperature: 1,
	});
	return (data.samples ?? []).filter(
		(s) =>
			s.anchor_name &&
			s.object_name &&
			s.placement &&
			s.anchor_description &&
			s.object_description,
	);
}
