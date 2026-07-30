# Placement — model output (example)

The reply is forced through OpenRouter structured outputs, so it is always a single JSON
object matching the schema `pipeline/place.mjs` sends with the request:

```json
"response_format": {
  "type": "json_schema",
  "json_schema": {
    "name": "placement",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "reasoning": { "type": "string" },
        "scale": { "type": "number" },
        "yaw_degrees": { "type": "number" },
        "pitch_degrees": { "type": "number" },
        "roll_degrees": { "type": "number" },
        "center": { "type": "array", "items": { "type": "number" } },
        "contact": { "type": "string", "enum": ["rest", "lean", "attach", "embed", "drape", "none"] },
        "contact_direction": { "type": "array", "items": { "type": "number" } },
        "embed_fraction": { "type": "number" }
      },
      "required": [
        "reasoning", "scale", "yaw_degrees", "pitch_degrees", "roll_degrees", "center",
        "contact", "contact_direction", "embed_fraction"
      ],
      "additionalProperties": false
    }
  }
}
```

The last three fields are the model's classification of its own placement, consumed by the
physics pass (`pipeline/physics.mjs`) that runs between the answer and the bake: `contact`
picks the refinement mode (settle under gravity, seat a lean, close a mount's gap, bury an
embed, or leave a float alone), `contact_direction` is the approach direction for
lean/attach/embed (`[0, 0, 0]` = infer the shortest way to A), and `embed_fraction` is how
much of B ends up inside A when embedding. For the bath-brush lean below, a sensible
classification is `contact: "lean"`, `contact_direction: [0, 0, -1]` (pressing toward the
tub), `embed_fraction: 0`.

A real answer to the prompt in [input.md](input.md), produced by the default model — note
the rotation doing genuine work: the phrase leans the brush, and yaw + roll stand it up
15° off vertical against the tub:

```json
{
  "reasoning": "The long object (B) needs to be placed upright on the floor (-Y) while leaning against the front face (+Z) of the tub (A) with a slight left (-X) position. I chose a scale of 0.35 so the brush size roughly matches typical real-world dimensions relative to the bathtub. Using a 90-degree yaw aligns the head bristles towards the back (-Z, towards the tub) and faces its spine outward. A pitch of 0 followed by a roll of -105 degrees makes the brush stand vertically (-90) and then tips it 15 degrees inward toward the tub. I calculated the lowest corner of the brush to touch the ground exactly at A's Y_min (-0.189), yielding a center_Y of -0.019. Similarly, the center_Z of 0.350 ensures the uppermost bristles exactly rest on the rim of the bathtub's exterior face (at Z = 0.283) without intersecting. Finally, a center_X of -0.150 positions it slightly left of center along the front exterior surface.",
  "scale": 0.35,
  "yaw_degrees": 90,
  "pitch_degrees": 0,
  "roll_degrees": -105,
  "center": [-0.15, -0.019, 0.35]
}
```

## What the pipeline does with it

`run.mjs` validates the answer (finite positive scale, finite angles, a 3-vector centre, a
known contact, a 3-vector direction, a finite embed fraction — anything else is retried as
a bad roll) and turns it into world transforms directly: the answer already speaks A's
bbox-centred frame, which becomes the world frame when the anchor is baked with its
bounding-box centre at the origin.

- **scale** — `0.35`, applied to B as-is.
- **rotation** — the three angles compose as intrinsic yaw → pitch → roll (three.js Euler
  order `YXZ`) into one quaternion, here `[-0.561, 0.430, -0.561, 0.430]`.
- **position** — B's bounding-box centre must land on `center`. The transform applies to
  B's file coordinates, so B's own bbox-centre offset (scaled, rotated) is subtracted
  first; farm meshes are near bbox-centred already, so the position is effectively the
  centre itself: `[-0.15, -0.019, 0.349]`.

The placed transform then goes through the physics pass (`pipeline/physics.mjs`): the
anchor's actual triangles become a signed distance field, the placed mesh becomes a few
thousand surface samples, and B is moved — dropped, seated, leaned or buried according to
`contact` — into exact contact with the true surfaces. The pass returns a corrected
position and rotation (never scale), plus a report of how far it moved B and what gap
remained, which lands in the placement log and in `metadata.physics`. Anything the pass
cannot do safely (no contact within range, a body that slid too far, a start buried deep
inside A) reverts to the model's transforms with a flag.

The anchor's transform is just its bounding-box centre moved to the origin (identity
rotation, unit scale). Both transforms are baked into the GLBs as a single new root node
named `placement`, so the files alone carry the pose; metadata only records the measured
per-object and combined sizes plus the physics report.
