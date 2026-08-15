// The scene-editing benchmark: one indoor scene, one instruction, several LLMs.
//
// A run is a scene and a prompt. Every model under test is handed the same scene as a list of
// axis-aligned bounding boxes and the same instruction, and answers with an ordered program of
// scene-operation calls. Nothing about the scene is derived per model — the boxes are measured
// once, when the run is created, and stored — so the only variable in a comparison is the model.
// `prompt_hash` covers the exact prompt and response contract every model received; a result
// whose hash no longer matches its run was answered against an older manifest or contract.
//
//   data/scene-edits/<runId>/
//     run.json                 the scene it points at, the prompt, the measured boxes
//     <model-slug>/edit.json   the calls as given, and the final transforms they resolve to
//
// The scenes are 3D-FRONT rooms as MIDI packages them, read out of data/3d-front/ rather than
// copied in: a room is 5-13 MB and a run is a pointer. Each is a single `world` node with one
// child per object, every transform identity and the placement baked into the vertices,
// already centred on the origin with its longest axis at 1.9. There is no floor or wall
// geometry, so the floor is the lowest point in the scene.
//
// Coordinates reach the model as integers in scene units: the normalised frame multiplied by
// `unit_scale`, so a room spans roughly ±95 and a bed measures 88 x 50 x 75. Three decimal
// places of a scale nothing can name are a source of arithmetic mistakes and nothing else. The
// client divides by `unit_scale` again, and because it also recentres each object on the centre
// recorded here, the rounding cancels exactly for anything a model left alone.

import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { Box3, MathUtils, Quaternion, Vector3 } from "three";
import { FRONT3D_DIR, SCENE_EDITS_DIR } from "./paths.mjs";
import { nodeBounds, nodeTriangles, parseGLB } from "./glb.mjs";
import { slug, writeAtomic } from "./metadata.mjs";
import { chatJSON, chatJSONWithImages } from "./openrouter.mjs";
import { renderView } from "./render.mjs";
import { mapLimit, retry, widthOf } from "./limit.mjs";

const RUN_FILE = "run.json";
const EDIT_FILE = "edit.json";
const SCENE_INFO_FILE = "objects.json";
const VIEW_DIR = "objects";

/** Normalised units × this reach the model, so every coordinate it reads is a small integer. */
const UNIT_SCALE = 100;
const IDENTIFY_MODEL = "openai/gpt-5.6-luna";
const IDENTIFY_VIEWS = [0, 90, 180, 270];
const IDENTIFY_ELEVATION = 22;
const STRUCTURAL_IDS = new Map([
	["floorglb", "floor"],
	["ceilglb", "ceil"],
	["wallglb", "wall"],
	["othersglb", "others"],
]);
/** How far above the lowest point in a scene an object may sit and still count as on the floor. */
const FLOOR_TOLERANCE = 2;

// The same four the placement benchmark runs, so the two comparisons cover one roster.
const DEFAULT_MODELS = [
	"openai/gpt-5.6-sol-pro",
	"moonshotai/kimi-k3",
	"anthropic/claude-fable-5",
	"google/gemini-3.1-pro-preview",
];

/** Read lazily: this module is evaluated before an entry point has finished loading .env. */
export const models = () =>
	(process.env.SCENE_EDIT_MODELS ?? DEFAULT_MODELS.join(","))
		.split(",")
		.map((model) => model.trim())
		.filter(Boolean);

/** `openai/gpt-5.6-sol-pro` → `openai-gpt-5.6-sol-pro`, the folder a model writes to. */
export const modelSlug = (model) =>
	model
		.toLowerCase()
		.replace(/[^a-z0-9.+-]+/g, "-")
		.replace(/^-+|-+$/g, "");

const safeId = (id) =>
	path.basename(String(id ?? "")).replace(/[^a-zA-Z0-9._-]+/g, "_");
const runDir = (id) => path.join(SCENE_EDITS_DIR, safeId(id));
const sceneKey = (file) => slug(path.basename(file, path.extname(file)));
const sceneInfoDir = (file) => path.join(SCENE_EDITS_DIR, "_scenes", sceneKey(file));
const sceneInfoFile = (file) => path.join(sceneInfoDir(file), SCENE_INFO_FILE);

const su = (value) => Math.round(value * UNIT_SCALE);
const suArray = (vector) => vector.toArray().map(su);
const round = (value) => Number(value.toFixed(6));

// ── measuring a scene ────────────────────────────────────────────────────────

/** `Cabinet_Shelf_Desk_<uuid>_3.glb` → the 3D-FRONT super-category and room index. */
const OBJECT_NAME =
	/^(.+?)_[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}_(\d+)$/i;

function describe(name, order) {
	const sourceId = name.toLowerCase().replace(/[^a-z0-9]+/g, "");
	const structural = STRUCTURAL_IDS.get(sourceId);
	if (structural) {
		return {
			id: sourceId,
			label: sourceId,
			category: structural,
			instance: null,
			structural: true,
		};
	}

	const stem = name.replace(/\.(?:obj|glb)$/i, "");
	const match = OBJECT_NAME.exec(stem);
	const category = match?.[1] ?? stem;
	const instance = match?.[2] ?? String(order + 1);
	return {
		id: [slug(category), instance].filter(Boolean).join("_"),
		label: category.toLowerCase().replace(/_+/g, " "),
		category,
		instance,
		structural: false,
	};
}

export const listScenes = () =>
	(fs.existsSync(FRONT3D_DIR) ? fs.readdirSync(FRONT3D_DIR) : [])
		.filter((file) => /\.glb$/i.test(file))
		.sort((a, b) => a.localeCompare(b));

/**
 * Every object in a room, as the run manifest records it.
 *
 * Geometry and source identity are measured here. Semantic names, stable scene-local IDs and
 * facing axes are attached by `identifyScene`, which renders four world-axis views per object.
 * The structural `floorglb`, `ceilglb`, `wallglb` and `othersglb` IDs bypass identification and
 * are preserved exactly.
 */
function describeScene(file, target) {
	const parts = nodeBounds(parseGLB(fs.readFileSync(target))).map((part, order) => {
		const description = describe(part.name, order);
		return {
			...part,
			...description,
			source_id: description.id,
			box: new Box3(
				new Vector3().fromArray(part.min),
				new Vector3().fromArray(part.max),
			),
		};
	});
	if (!parts.length) throw new Error(`${file} has no objects`);

	const bounds = new Box3();
	for (const part of parts) bounds.union(part.box);
	const floor = su(bounds.min.y);

	const shared = {};
	for (const part of parts)
		if (!part.structural)
			shared[part.category] = (shared[part.category] ?? 0) + 1;

	return {
		file,
		unit_scale: UNIT_SCALE,
		bounds: { min: suArray(bounds.min), max: suArray(bounds.max) },
		floor_y: floor,
		objects: parts.map((part) => ({
			id: part.id,
			source_id: part.source_id,
			label:
				!part.structural && shared[part.category] > 1
					? `${part.label} ${part.instance}`
					: part.label,
			category: part.category,
			node: part.name,
			index: part.index,
			center: suArray(part.box.getCenter(new Vector3())),
			size: suArray(part.box.getSize(new Vector3())),
			on_floor: su(part.box.min.y) - floor <= FLOOR_TOLERANCE,
			structural: part.structural,
			facing_axis: null,
		})),
	};
}

const measured = new Map();

/** Memoised on the file's mtime: a room is a dozen megabytes and never changes under us. */
export function measure(file) {
	const name = path.basename(String(file ?? ""));
	const target = path.join(FRONT3D_DIR, name);
	if (!name || !fs.existsSync(target))
		throw new Error(`no scene named ${name || "(nothing)"}`);
	const key = `${name}:${fs.statSync(target).mtimeMs}`;
	if (!measured.has(key)) measured.set(key, describeScene(name, target));
	return measured.get(key);
}

// ── identifying the objects ──────────────────────────────────────────────────

const AXIS_COLORS = { x: [220, 60, 60], y: [45, 180, 75], z: [45, 105, 225] };
const FACING_AXES = ["+X", "-X", "+Z", "-Z"];

function axisTriangles(center, size) {
	const length = Math.max(size.x, size.y, size.z) * 0.7;
	const width = Math.max(length * 0.025, 0.003);
	const prism = (axis) => {
		const min = center.clone().addScalar(-width);
		const max = center.clone().addScalar(width);
		max[axis] += length;
		const [x0, y0, z0] = min.toArray();
		const [x1, y1, z1] = max.toArray();
		const corners = [
			[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
			[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
		];
		const faces = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5];
		return new Float64Array(faces.flatMap((index) => corners[index]));
	};
	return [
		{ triangles: prism("x"), base: AXIS_COLORS.x },
		{ triangles: prism("y"), base: AXIS_COLORS.y },
		{ triangles: prism("z"), base: AXIS_COLORS.z },
	];
}

const IDENTIFY_SYSTEM = `You identify one isolated 3D indoor-scene object from four renders taken 90 degrees apart around world +Y.

The images are ordered camera yaw 0, 90, 180 and 270 degrees. Every camera is slightly above the object and looks slightly downward. World axes are drawn from the object's centre: +X is red, +Y is green and +Z is blue.

Return a concise everyday object name, a lowercase snake_case id based on that name, and the world horizontal axis the front of the object points along. Front means the direction a person using the object faces, or the visually functional front (chair seat faces out, desk user side, cabinet doors, television screen, bed head-to-foot direction). Choose exactly one of +X, -X, +Z or -Z. If the object is rotationally symmetric or genuinely has no meaningful front, return null for front_axis.`;

const IDENTIFY_SCHEMA = {
	type: "object",
	properties: {
		name: { type: "string" },
		id: { type: "string" },
		front_axis: { type: ["string", "null"], enum: [...FACING_AXES, null] },
	},
	required: ["name", "id", "front_axis"],
	additionalProperties: false,
};

function readSceneInfo(file) {
	const target = sceneInfoFile(file);
	return fs.existsSync(target) ? JSON.parse(fs.readFileSync(target, "utf8")) : null;
}

function semanticId(value, fallback) {
	const id = slug(String(value ?? ""));
	return id === "object" ? fallback : id;
}

export function assignUniqueIds(objects, answers) {
	const bases = objects.map((object) => {
		if (object.structural) return object.id;
		const answer = answers.get(object.node);
		if (!answer) throw new Error(`${object.node} was not identified`);
		return semanticId(answer.id || answer.name, object.id);
	});
	const totals = bases.reduce((counts, base) => ({ ...counts, [base]: (counts[base] ?? 0) + 1 }), {});
	const seen = {};

	return objects.map((object, index) => {
		if (object.structural) return object;
		const answer = answers.get(object.node);
		const base = bases[index];
		seen[base] = (seen[base] ?? 0) + 1;
		return {
			...object,
			id: totals[base] === 1 ? base : `${base}_${seen[base]}`,
			label: String(answer.name).trim(),
			facing_axis: answer.front_axis,
		};
	});
}

async function identifyObject(glb, object, directory, force) {
	const folder = path.join(directory, VIEW_DIR, String(object.index).padStart(3, "0"));
	fs.mkdirSync(folder, { recursive: true });
	const triangles = nodeTriangles(glb, object.index);
	const box = new Box3(
		new Vector3().fromArray(object.center).divideScalar(UNIT_SCALE).sub(new Vector3().fromArray(object.size).divideScalar(UNIT_SCALE * 2)),
		new Vector3().fromArray(object.center).divideScalar(UNIT_SCALE).add(new Vector3().fromArray(object.size).divideScalar(UNIT_SCALE * 2)),
	);
	const center = box.getCenter(new Vector3());
	const size = box.getSize(new Vector3());
	const views = [];
	for (const yaw of IDENTIFY_VIEWS) {
		const target = path.join(folder, `yaw-${String(yaw).padStart(3, "0")}.png`);
		if (force || !fs.existsSync(target)) {
			const image = await renderView(
				[{ triangles }, ...axisTriangles(center, size)],
				{ size: 768, view: { azimuth_deg: yaw, elevation_deg: IDENTIFY_ELEVATION } },
			);
			writeAtomic(target, image);
		}
		views.push(fs.readFileSync(target));
	}

	const reply = await retry(() => chatJSONWithImages({
		model: process.env.SCENE_IDENTIFY_MODEL ?? IDENTIFY_MODEL,
		system: IDENTIFY_SYSTEM,
		user: `Identify this object. Its source node is ${JSON.stringify(object.node)}. Images 1-4 are yaw ${IDENTIFY_VIEWS.join(", ")} degrees respectively.`,
		images: views.map((data) => ({ mimeType: "image/png", data })),
		name: "scene_object",
		schema: IDENTIFY_SCHEMA,
		reasoning: { effort: "none" },
	}));
	const answer = reply.data;
	if (!String(answer?.name ?? "").trim()) throw new Error(`${object.node}: empty object name`);
	if (answer.front_axis !== null && !FACING_AXES.includes(answer.front_axis)) {
		throw new Error(`${object.node}: invalid front axis ${JSON.stringify(answer.front_axis)}`);
	}
	return { ...answer, model_used: reply.model, usage: reply.usage, views: IDENTIFY_VIEWS.map((yaw) => `${VIEW_DIR}/${String(object.index).padStart(3, "0")}/yaw-${String(yaw).padStart(3, "0")}.png`) };
}

/** Render and identify every non-structural object once, then persist the semantic scene. */
export async function identifyScene(file, { force = false, log = () => {} } = {}) {
	const scene = structuredClone(measure(file));
	const target = path.join(FRONT3D_DIR, scene.file);
	const stat = fs.statSync(target);
	const previous = readSceneInfo(scene.file);
	if (!force && previous?.source_mtime_ms === stat.mtimeMs) return previous;

	const directory = sceneInfoDir(scene.file);
	fs.mkdirSync(directory, { recursive: true });
	const glb = parseGLB(fs.readFileSync(target));
	const candidates = scene.objects.filter((object) => !object.structural);
	const previousByNode = force
		? new Map()
		: new Map((previous?.objects ?? []).filter((object) => object.identification).map((object) => [object.node, object.identification]));
	let finished = 0;
	log(`rendering and identifying 0/${candidates.length} objects`);
	const identified = await mapLimit(
		candidates,
		widthOf("SCENE_IDENTIFY_CONCURRENCY", 4),
		async (object) => {
			const answer = previousByNode.get(object.node) ?? await identifyObject(glb, object, directory, force);
			finished++;
			log(`rendering and identifying ${finished}/${candidates.length} objects`);
			return answer;
		},
	);
	const answers = new Map(candidates.map((object, index) => [object.node, identified[index]]));
	const record = {
		version: 1,
		file: scene.file,
		source_mtime_ms: stat.mtimeMs,
		created_at: new Date().toISOString(),
		model: process.env.SCENE_IDENTIFY_MODEL ?? IDENTIFY_MODEL,
		view: { yaw_degrees: IDENTIFY_VIEWS, elevation_degrees: IDENTIFY_ELEVATION, axes: AXIS_COLORS },
		objects: assignUniqueIds(scene.objects, answers).map((object) => ({
			...object,
			identification: object.structural ? null : answers.get(object.node),
		})),
	};
	writeAtomic(sceneInfoFile(scene.file), `${JSON.stringify(record, null, 2)}\n`);
	return record;
}

function identifiedScene(file) {
	const scene = measure(file);
	const info = readSceneInfo(scene.file);
	if (!info) throw new Error(`${scene.file} has not been identified yet`);
	return { ...scene, objects: structuredClone(info.objects) };
}

// ── the prompt ───────────────────────────────────────────────────────────────

const SYSTEM = `You are editing a 3D indoor scene. You are given every object in it as an axis-aligned bounding box, and one instruction. Return an ordered array of calls to the scene API that carries the instruction out.

Coordinates are right-handed with +Y up. Object-facing directions are explicit world axes: +X, -X, +Z or -Z. A rotate_object call changes that facing by its relative yaw.

Positions are in scene units. The scene is centred on the origin, so every coordinate is already relative to (0,0,0), and one unit is a hundredth of the scene's longest axis. Each object is given as its bounding-box centre and the box's extents, so its box spans centre ± size/2 and its lowest point is centre.y - size.y/2.

The scene API has exactly these calls:
- move_object(id, center): move an object's bounding-box centre to the absolute [x, y, z] position.
- rotate_object(id, yaw_degrees): rotate an object by this many degrees about +Y using the right-hand rule. The angle is a change from its current orientation, not an absolute heading.
- scale_object(id, factor): multiply an object's size uniformly by a positive factor.
- delete_object(id): remove an object from the scene.

Return only the calls needed, in execution order. Each array entry has the four call names as keys. Set exactly one key to that call's arguments object and the other three keys to null. Do not put arguments belonging to one call under another key. Use an id from the object list exactly as written. You may call several operations on the same object; each call observes the effects of earlier calls. Do not emit no-op calls.

Rotation and scale are about the object's own bounding-box centre, so they do not move that centre. Turning an object sweeps its footprint through the space around it, and scaling one grows it in every direction at once. If another call must reposition it to keep it seated or avoid a collision, include that move explicitly.

The scene holds furniture only — there is no floor or wall geometry. An object marked as on the floor rests on the floor plane and must still rest on it after the edit, so centre.y = floor + final_size.y/2, unless the instruction says otherwise. An object that is not on the floor — a ceiling light, something standing on a surface — keeps the clearance it has.

A moved object must not end up inside another object it was not already touching, and must stay within the scene's extents. Contact is deliberate: "beside", "against" and "next to" mean touching or nearly touching, not a long way off.

You cannot add objects. The only objects available are the ones listed. The structural IDs floorglb, ceilglb, wallglb and othersglb are fixed room geometry: never move, rotate, scale or delete them.`;

const CALL_SCHEMA = {
	type: "object",
	properties: {
		move_object: {
			type: ["object", "null"],
			properties: {
				id: { type: "string" },
				center: {
					type: "array",
					items: { type: "number" },
					minItems: 3,
					maxItems: 3,
				},
			},
			required: ["id", "center"],
			additionalProperties: false,
		},
		rotate_object: {
			type: ["object", "null"],
			properties: {
				id: { type: "string" },
				yaw_degrees: { type: "number" },
			},
			required: ["id", "yaw_degrees"],
			additionalProperties: false,
		},
		scale_object: {
			type: ["object", "null"],
			properties: {
				id: { type: "string" },
				factor: { type: "number" },
			},
			required: ["id", "factor"],
			additionalProperties: false,
		},
		delete_object: {
			type: ["object", "null"],
			properties: { id: { type: "string" } },
			required: ["id"],
			additionalProperties: false,
		},
	},
	required: ["move_object", "rotate_object", "scale_object", "delete_object"],
	additionalProperties: false,
};

const SCHEMA = {
	type: "object",
	properties: {
		reasoning: { type: "string" },
		calls: { type: "array", items: CALL_SCHEMA },
	},
	required: ["reasoning", "calls"],
	additionalProperties: false,
};

/** The exact two messages every model under test is sent, built from the manifest alone. */
export function buildPrompt(run) {
	const { objects, scene, prompt } = run;

	const columns = [
		["id", (object) => object.id],
		["name", (object) => object.label],
		["centre", (object) => object.center.join(", ")],
		["size", (object) => object.size.join(", ")],
		["on floor", (object) => (object.on_floor ? "yes" : "no")],
		["front axis", (object) => object.facing_axis ?? "none"],
	];
	const widths = columns.map(([head, read]) =>
		Math.max(head.length, ...objects.map((object) => read(object).length)),
	);
	const row = (cells) =>
		cells
			.map((cell, index) => cell.padEnd(widths[index]))
			.join("   ")
			.trimEnd();
	const extents = ["x", "y", "z"]
		.map(
			(axis, index) =>
				`${axis} ${scene.bounds.min[index]}..${scene.bounds.max[index]}`,
		)
		.join(", ");

	return {
		system: SYSTEM,
		user: [
			`Scene: ${scene.file.replace(/\.glb$/i, "")}, ${objects.length} objects.`,
			`Floor plane: y = ${scene.floor_y}. Extents: ${extents}.`,
			"",
			row(columns.map(([head]) => head)),
			...objects.map((object) =>
				row(columns.map(([, read]) => read(object))),
			),
			"",
			`Instruction: "${prompt}"`,
		].join("\n"),
	};
}

export function promptHash(run) {
	const { system, user } = buildPrompt(run);
	return createHash("sha256")
		.update(`${system}\n\n${user}`)
		.digest("hex")
		.slice(0, 32);
}

// ── runs ─────────────────────────────────────────────────────────────────────

/**
 * `prompt_hash` is derived here rather than stored, because it is a property of the manifest
 * and not a fact about it. Relabelling an object by hand — which is how a prompt about "the
 * wardrobe" becomes answerable — changes the text every model is sent, and a hash written into
 * the file would go on claiming the old text. Computed on read, the results answered against
 * the old wording go stale the moment the file is saved, which is the point of having it.
 */
export function readRun(id) {
	const file = path.join(runDir(id), RUN_FILE);
	if (!fs.existsSync(file)) throw new Error(`no run named ${safeId(id)}`);
	const run = JSON.parse(fs.readFileSync(file, "utf8"));
	return { ...run, prompt_hash: promptHash(run) };
}

export const listRuns = () =>
	(fs.existsSync(SCENE_EDITS_DIR)
		? fs.readdirSync(SCENE_EDITS_DIR, { withFileTypes: true })
		: []
	)
		.filter(
			(entry) =>
				entry.isDirectory() &&
				fs.existsSync(path.join(SCENE_EDITS_DIR, entry.name, RUN_FILE)),
		)
		.map((entry) => readRun(entry.name))
		.sort((a, b) => a.id.localeCompare(b.id));

/** Results from the field-based contract are kept on disk but hidden from this call-based UI. */
const isCallResult = (result) => Array.isArray(result.answer?.calls);

/** Every model that has answered this run under the current call contract. */
export function readResults(id) {
	const dir = runDir(id);
	return (
		fs.existsSync(dir) ? fs.readdirSync(dir, { withFileTypes: true }) : []
	)
		.filter(
			(entry) =>
				entry.isDirectory() &&
				fs.existsSync(path.join(dir, entry.name, EDIT_FILE)),
		)
		.map((entry) => ({
			slug: entry.name,
			...JSON.parse(
				fs.readFileSync(path.join(dir, entry.name, EDIT_FILE), "utf8"),
			),
		}))
		.filter(isCallResult)
		.sort((a, b) => a.model.localeCompare(b.model));
}

/** Scene-list metadata includes whether the one-time vision pass has completed. */
export function sceneSummary(file) {
	const scene = measure(file);
	const info = readSceneInfo(scene.file);
	return {
		file: scene.file,
		identified: !!info,
		objects: (info?.objects ?? scene.objects).map(({ id, label, size, structural, facing_axis }) => ({
			id,
			label,
			size,
			structural,
			facing_axis,
		})),
	};
}

/** Named after its room, since a run exists before there is an instruction to name it after. */
function freeId(file) {
	const base = slug(file.replace(/\.glb$/i, ""));
	let id = base;
	for (let n = 2; fs.existsSync(runDir(id)); n++) id = `${base}-${n}`;
	return id;
}

/**
 * A run starts from an already identified scene. The instruction arrives through `setPrompt`
 * after the semantic object names and facing axes have been copied into the immutable run
 * manifest, keeping every model in that comparison on exactly the same object vocabulary.
 */
export function createRun({ scene, prompt, id }) {
	const text = String(prompt ?? "").trim();

	const { file, unit_scale, bounds, floor_y, objects } = identifiedScene(scene);
	const runId = id ? safeId(id) : freeId(file);
	const dir = runDir(runId);
	if (fs.existsSync(dir)) throw new Error(`${runId} already exists`);

	const run = {
		id: runId,
		created_at: new Date().toISOString(),
		prompt: text,
		scene: { file, unit_scale, bounds, floor_y },
		objects,
	};

	fs.mkdirSync(dir, { recursive: true });
	writeAtomic(path.join(dir, RUN_FILE), `${JSON.stringify(run, null, 2)}\n`);
	return { ...run, prompt_hash: promptHash(run) };
}

/**
 * Read, change, write back. Nothing is written if `change` throws, so a rejected edit leaves the
 * manifest exactly as it was.
 */
function updateRun(id, change) {
	const file = path.join(runDir(id), RUN_FILE);
	if (!fs.existsSync(file)) throw new Error(`no run named ${safeId(id)}`);
	const run = JSON.parse(fs.readFileSync(file, "utf8"));
	change(run);
	writeAtomic(file, `${JSON.stringify(run, null, 2)}\n`);
	return { ...run, prompt_hash: promptHash(run) };
}

export const setPrompt = (id, prompt) =>
	updateRun(id, (run) => {
		run.prompt = String(prompt ?? "").trim();
	});

/**
 * Renames objects. The seeded labels are 3D-FRONT super-categories, so one room can hold three
 * `cabinet shelf desk`, and an instruction about "the wardrobe" only means something once one of
 * them is called that. A rename goes into the text models are sent, so — like the instruction
 * itself — it takes every answer already given stale.
 */
export const setLabels = (id, labels) =>
	updateRun(id, (run) => {
		for (const [objectId, label] of Object.entries(labels ?? {})) {
			const object = run.objects.find((entry) => entry.id === objectId);
			if (!object) throw new Error(`${run.id} has no object ${objectId}`);
			if (object.structural) throw new Error(`${objectId} is structural and cannot be renamed`);
			const text = String(label ?? "").trim();
			if (!text) throw new Error(`${objectId} needs a label`);
			object.label = text;
		}
	});

// ── asking a model ───────────────────────────────────────────────────────────

const UP = new Vector3(0, 1, 0);

/**
 * Executes a model's scene calls in order and collapses them to one final pose per object, in
 * normalised units, for the viewer. Calls on the same object compose; a rejected call changes
 * nothing, and later calls still execute against the last valid state.
 */
export function resolveCalls(run, calls) {
	const units = (values) =>
		values.map((value) => round(value / run.scene.unit_scale));
	const states = new Map(
		run.objects.map((object) => [
			object.id,
			{
				origin: units(object.center),
				position: units(object.center),
				yaw: 0,
				scale: 1,
				deleted: false,
				touched: false,
			},
		]),
	);
	const rejected = [];

	for (const [index, call] of calls.entries()) {
		const active = ["move_object", "rotate_object", "scale_object", "delete_object"]
			.filter((name) => call?.[name] !== null && call?.[name] !== undefined);
		if (active.length !== 1) {
			rejected.push({
				index,
				name: null,
				id: null,
				why: `a call must select exactly one operation, found ${active.length}`,
			});
			continue;
		}
		const name = active[0];
		const args = call[name];
		const id = args?.id;
		const state = states.get(id);
		const reject = (why) => rejected.push({ index, name, id: id ?? null, why });

		if (!state) {
			reject("no object by that id");
			continue;
		}
		if (state.deleted) {
			reject("the object was already deleted");
			continue;
		}

		switch (name) {
			case "move_object": {
				const center = args.center;
				if (!Array.isArray(center) || center.length !== 3 || !center.every(Number.isFinite)) {
					reject(`center must be three finite numbers: ${JSON.stringify(center)}`);
					break;
				}
				state.position = units(center);
				state.touched = true;
				break;
			}
			case "rotate_object": {
				const yaw = args.yaw_degrees;
				if (!Number.isFinite(yaw)) {
					reject(`yaw_degrees must be finite: ${JSON.stringify(yaw)}`);
					break;
				}
				state.yaw += yaw;
				state.touched = true;
				break;
			}
			case "scale_object": {
				const factor = args.factor;
				if (!Number.isFinite(factor) || factor <= 0) {
					reject(`factor must be positive: ${JSON.stringify(factor)}`);
					break;
				}
				state.scale *= factor;
				state.touched = true;
				break;
			}
			case "delete_object":
				state.deleted = true;
				state.touched = true;
				break;
			default:
				reject(`unknown call ${JSON.stringify(name)}`);
		}
	}

	const resolved = [];
	for (const [id, state] of states) {
		if (!state.touched) continue;
		if (state.deleted) {
			resolved.push({ id, action: "delete", origin: state.origin });
			continue;
		}
		resolved.push({
			id,
			action: "transform",
			origin: state.origin,
			position: state.position,
			rotation: new Quaternion()
				.setFromAxisAngle(UP, MathUtils.degToRad(state.yaw))
				.toArray()
				.map(round),
			scale: [state.scale, state.scale, state.scale].map(round),
		});
	}
	return { resolved, rejected };
}

/** One model's answer to one run, reused off disk unless `force` is set. */
export async function askModel(run, model, { force = false } = {}) {
	if (!run.prompt) throw new Error(`${run.id} has no instruction to ask about`);
	const dir = path.join(runDir(run.id), modelSlug(model));
	const file = path.join(dir, EDIT_FILE);
	if (!force && fs.existsSync(file)) {
		const previous = JSON.parse(fs.readFileSync(file, "utf8"));
		if (isCallResult(previous) && previous.prompt_hash === run.prompt_hash)
			return previous;
	}

	const { system, user } = buildPrompt(run);
	const started = Date.now();
	const answer = await retry(async () => {
		const reply = await chatJSON({
			model,
			system,
			user,
			name: "scene_edit",
			schema: SCHEMA,
		});
		if (!Array.isArray(reply.data?.calls))
			throw new Error(
				`no calls array: ${JSON.stringify(reply.data).slice(0, 300)}`,
			);
		return reply;
	});

	const { resolved, rejected } = resolveCalls(run, answer.data.calls);
	const result = {
		run: run.id,
		model,
		model_used: answer.model,
		prompt_hash: run.prompt_hash,
		created_at: new Date().toISOString(),
		latency_ms: Date.now() - started,
		usage: answer.usage,
		answer: {
			reasoning: answer.data.reasoning ?? "",
			calls: answer.data.calls,
		},
		resolved,
		rejected,
	};

	fs.mkdirSync(dir, { recursive: true });
	writeAtomic(file, `${JSON.stringify(result, null, 2)}\n`);
	return result;
}
