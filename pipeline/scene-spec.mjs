// The one definition of the six placement categories and the settings scenes are staged in.
//
// Both halves of seeded generation read this file: `objaverse-pool.mjs` tags every asset with
// the categories it can serve (as the anchor that receives, and as the object that is
// placed), and `generate-scenes.mjs` builds its per-category prompts from the same wording.
// Keeping them in one place means the tagger and the generator can never drift apart on what
// a category requires.
//
// Each category carries:
//   weight    its share of a run's total — rigid and soft are deliberately over-represented
//   anchor    what the receiving asset's geometry must afford
//   placed    what the placed object must be
//   simple / complex   each half's relation list. Relations are pure spatial primitives —
//             "on top of", "inside", "attached to" — and nothing more. Which side, which
//             part, how deep, what orientation: all of that is the model's per-pairing
//             choice, steered by the dealt detail level (bare / position / part), because
//             only the model knows which parts the chosen asset actually has.
//   examples  one phrase per detail level, shown to the model for its assigned level

export const CATEGORIES = [
  {
    id: 'rigid',
    label: 'Rigid pose placement',
    weight: 2,
    brief:
      'The placed object keeps its shape and simply takes a position, orientation and size ' +
      'against an external surface of the anchor — from any direction.',
    anchor: 'stable external surfaces something can rest on, lean against, or hang from',
    placed: 'a rigid object sized to rest, lean or balance stably',
    simple: { relations: ['on top of', 'leaning against'] },
    complex: { relations: ['on top of', 'leaning against', 'hanging on', 'under'] },
    examples: {
      bare: 'lamp on top of the dresser',
      position: 'cup sitting on top of the kitchen island in the middle',
      part: 'cup on top of the kitchen island next to the sink cutout',
    },
  },
  {
    id: 'soft',
    label: 'Soft-body deformation',
    weight: 1.5,
    brief:
      'The placed object is soft and conforms to the anchor where they meet — it drapes, ' +
      'folds and hangs rather than keeping a rigid pose.',
    anchor: 'a raised form fabric can drape over — a back, rail, arm, beam, branch or edge standing proud of the body',
    placed: 'soft and drapeable — clothing, blankets, towels, flags, nets, rope',
    simple: { relations: ['laid over', 'hanging over'] },
    complex: { relations: ['laid over', 'hanging over', 'wrapped around'] },
    examples: {
      bare: 'towel laid over the couch',
      position: 'towel laid over the couch on the left side',
      part: 'jacket hanging over the back of the chair',
    },
  },
  {
    id: 'penetrative',
    label: 'Penetrative embedding',
    weight: 1,
    brief:
      "The placed object breaches the anchor's surface where no opening existed, ending " +
      'partially sunk into the material.',
    anchor:
      'a penetrable material — anything predominantly wood, cork, foam, soil, sand, wax or ' +
      'straw, so tables, crates, fences, stumps and boards all qualify',
    placed: 'rigid with a point, blade, spike or stake',
    simple: { relations: ['inserted into', 'pushed into'] },
    complex: { relations: ['inserted into', 'pushed into'] },
    examples: {
      bare: 'knife inserted into the cutting board',
      position: 'knife inserted into the cutting board in the middle',
      part: 'nail pushed into the leg of the workbench',
    },
  },
  {
    id: 'containment',
    label: 'Containment / fitted insertion',
    weight: 1,
    brief:
      "The final state is defined by the anchor's hollow interior, cavity, slot or opening " +
      'rather than a support surface — anywhere from loosely tossed in to snugly fitted.',
    anchor: 'an accessible hollow, cavity, slot, rack or opening',
    placed: "compact and rigid, sized to fit the anchor's cavity, slot or opening",
    simple: { relations: ['inside', 'dropped into'] },
    complex: { relations: ['inside', 'dropped into', 'slid into', 'plugged into', 'snapped onto'] },
    examples: {
      bare: 'keys dropped into the bowl',
      position: 'book slid into the bookcase on the right side',
      part: 'book slid into the middle shelf of the bookcase',
    },
  },
  {
    id: 'bonded',
    label: 'Bonded attachment',
    weight: 1,
    brief:
      'The placed object is registered flush against a face of the anchor at any ' +
      'orientation — fastened, adhered, pinned or clipped, with no settling or balancing. ' +
      'The object stays rigid; wrapping or conforming belongs to soft-body instead.',
    anchor: 'a clean face to affix to — doors, panels, boards, appliances, cabinets, hulls',
    placed: 'thin, light and flat-backed — magnets, notes, stickers, signs, decals, hooks',
    simple: { relations: ['attached to', 'taped to'] },
    complex: { relations: ['attached to', 'taped to', 'glued to', 'pinned to', 'clipped to'] },
    examples: {
      bare: 'magnet attached to the fridge',
      position: 'note taped to the cabinet on the left side',
      part: 'magnet attached to the freezer door of the fridge',
    },
  },
  {
    id: 'noncontact',
    label: 'Non-contact relational placement',
    weight: 1,
    brief:
      'The placed object touches nothing — its position in free space is defined purely by ' +
      'a spatial relationship to the anchor.',
    anchor:
      'any distinct free-standing object — the placed object only needs a spatial reference ' +
      'to hover or float around',
    placed: 'plausibly airborne — drones, balloons, birds, insects, kites, bubbles, embers',
    simple: { relations: ['floating above', 'floating next to'] },
    complex: {
      relations: [
        'floating above',
        'floating next to',
        'floating in front of',
        'floating behind',
        'floating below',
        'hovering above',
      ],
    },
    examples: {
      bare: 'balloon floating above the table',
      position: 'balloon floating above the table near the left edge',
      part: 'butterfly floating above the headboard of the bed',
    },
  },
];

export const CATEGORY_IDS = CATEGORIES.map((category) => category.id);

/** What the pool tagger treats as unusable for any role, in the words the prompt uses. */
export const JUNK =
  'an entire scene, room, building interior or terrain chunk, a collection of separate ' +
  'objects, a flat image or billboard, or an unidentifiable fragment';

// Where scenes are staged. Dealt round-robin per category, so 50k samples spread evenly
// rather than clustering on whichever settings the model likes. Broad on purpose: the mix
// runs mundane to fantastic, because the asset pool (game-artist content) covers both.
export const CONTEXTS = [
  'kitchen', 'walk-in closet', 'mudroom', 'bathroom', 'garage', 'pantry', 'attic',
  'basement workshop', 'laundry room', 'home office', 'nursery', 'sunroom', 'garden shed',
  'rooftop terrace', 'campsite', 'kids playroom', 'pillow fort', 'treehouse hideout',
  'backyard patio', 'halloween porch',
  'coffee shop', 'barber shop', 'hardware store', 'florist stall', 'bakery', 'butcher shop',
  'mechanic garage', 'veterinary clinic', 'pharmacy', 'bookstore', 'record store',
  'thrift store', 'sushi bar', 'ramen shop', 'food truck interior', 'ice cream parlor',
  'flea market stall', 'yard sale', 'christmas market stall', 'arcade prize counter',
  'school classroom', 'chemistry laboratory', 'gym locker room', 'library reading room',
  'museum gallery', 'art studio', 'theater backstage', 'recording studio', 'fire station',
  'science fair', 'pottery studio', 'glassblowing workshop', 'clockmaker workshop',
  'taxidermy shop', 'magic shop', 'apothecary',
  'factory floor', 'warehouse loading dock', 'shipping container yard', 'sawmill',
  'foundry', 'brewery', 'greenhouse nursery', 'construction site', 'fishing dock',
  'farm barn', 'orchard', 'vineyard', 'community garden', 'chicken coop', 'beekeeping yard',
  'koi pond garden', 'zen rock garden', 'beach boardwalk', 'ski lodge', 'desert oasis',
  'retro arcade', 'wild west saloon', 'speakeasy bar', 'circus tent', 'puppet theater',
  'camper van interior', 'submarine interior', 'lighthouse interior', 'medieval castle armory',
  'viking longhouse', 'samurai dojo', 'pirate ship deck', 'haunted mansion', 'necromancer crypt',
  'wizard alchemy tower', 'dwarven forge', 'elven treehouse', 'dragon hoard cave',
  'steampunk airship', 'cyberpunk street market', 'mars colony habitat', 'lunar base garage',
  'starship engine room', 'robot repair bay', 'post-apocalyptic bunker', 'space station module',
  'candy kingdom', 'candy factory', 'toy assembly line', 'mario platformer level', 'minecraft',
  'advanced alien civilization', 'gnome burrow', 'jungle temple ruins',
];
