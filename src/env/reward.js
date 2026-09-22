// What the agent is paid for.
//
// Hurting the others, killing them, and staying alive — plus a smaller set of
// terms about actually getting somewhere, because a worm wedged in a hole and a
// worm biding its time look identical to a scoreboard.
//
// Nothing here forbids anything. A worm that fires a bazooka at its own feet
// loses health and is charged for it in the same currency as being shot, so
// "do not blow yourself up" is something the agent works out rather than a rule
// someone has to write and tune. In the earlier repository self-inflicted
// damage was 41-56% of all damage taken, so there is a great deal to work out.
//
// With three worms in one world, "the other one lost health" is not a signal:
// two of them can be fighting while the third watches. Every number below comes
// from the engine's own attribution — who hit whom, for how much.

export const DEFAULT_WEIGHTS = {
  // Dealing damage pays twice what taking it costs, and a kill is worth twice a
  // death. Not for taste — for arithmetic.
  //
  // Everyone here is the same policy, so a fight is symmetric: over many
  // episodes each worm deals as much as it takes and kills as often as it dies.
  // Weight those equally and the whole of combat sums to exactly zero, while
  // the worm's own grenades still cost it something. The best strategy then is
  // to never fire, and in a run at these settings that is precisely what was
  // learned: after 1.2M steps deaths had fallen from 3.0 an episode to 0.01 and
  // damage dealt to nothing at all. It had worked out that the safest thing in
  // Liero is to stand still.
  //
  // Asymmetric, fighting has positive expected value and hurting yourself is
  // still charged in full.
  damageDealt: 1 / 100,
  damageTaken: 0.5 / 100,
  kill: 4,
  death: 2,
  // On top of `death`, for a death nobody else caused: its own grenade, its
  // own rocket, a fall. At the weights above a life traded for a kill nets
  // +2, and in the first cluster run 44% of all deaths were self-inflicted —
  // the arithmetic does not mind a worm that blows itself up on the way to
  // one kill. Zero until an evaluation says what it should be: it is a knob
  // for that experiment (`--suicide-cost`), not a setting anyone has measured.
  suicide: 0,
  // Covering the map, as a share of it rather than a count of cells: the
  // community maps are up to twelve times the area of a stock one, and a flat
  // per-cell payment made wandering them worth more than any fight. Measured
  // on a run that had stopped fighting altogether: exploring paid 4.99 an
  // episode while a kill and the death that came with it netted 0.12.
  //
  // These are what covering the WHOLE map is worth. A typical episode sees a
  // few percent of it, so this is a nudge away from camping, and the stuck and
  // revisit terms are what actually punish sitting still.
  exploreMap: 3,
  // Coming back to somewhere it was recently. Pacing a short loop trips this
  // every step and nothing else notices it.
  revisitMap: 3,
  // Per step spent unable to move. It has to hurt enough to be worth digging
  // out of, and not so much that dying becomes the cheaper way out of a hole.
  // At this weight, being stuck long enough to be worth dying over is about 200
  // steps — thirteen seconds — by which point the worm really has dropped out
  // of the match.
  stuck: 0.01,
  // Per pixel of progress toward a goal, when one is set, and for arriving.
  goalProgress: 0.02,
  reachedGoal: 2,
  // Pointing at somebody it could actually hit, per decision, scaled by how
  // centred the aim is. Aiming earns nothing by itself in this game — the
  // payoff arrives later as damage, if the shot lands — so a policy that
  // cannot aim never fires well enough to discover that aiming was the point.
  // This is the ladder up to that discovery, and it is deliberately small: the
  // reward for hitting somebody has to stay the reason to do it.
  // Per pixel closed on the nearest foe, and charged the same for backing off.
  // Over a whole episode this adds up to the distance between where a fight
  // started and where it ended, whatever route was taken — so there is no way
  // to earn it by pacing, and standing still earns nothing at all. It is the
  // first rung: a policy that cannot find anybody cannot learn to shoot them.
  approach: 0.001,
  // Back on. It was switched off on a measurement that turned out to be the
  // measurement's fault: the shot's *velocity* carries gravity, spread and in
  // most weapons the worm's own movement, and reading it said the aim was 180
  // degrees out. Read as the distance a shot actually travels in one tick, from
  // a worm standing still, with a weapon that has no spread and inherits no
  // speed, the view's angle matches where the shot goes to the degree.
  onTarget: 0.004,
  // And firing while lined up, which is the behaviour actually wanted. Worth
  // more than the aim alone, and it cannot be earned by standing and staring.
  aimedShot: 0.025,
};

/**
 * Getting somewhere is the whole score.
 *
 * The fighting terms are set to zero rather than removed, so the same reward
 * function runs and the same stats come out of it. A later stage turns them
 * back on by changing weights, not by taking a different path through the code.
 *
 * `stuck` stays: it is not a rung up to fighting, it is what stops a worm
 * sitting in a hole, and that is still true here. `exploreMap` and `revisitMap`
 * go, because they pay for covering ground rather than for arriving, and a worm
 * walking a straight line to a goal crosses its own trail whenever the route
 * doubles back.
 */
export const MOVEMENT_WEIGHTS = {
  ...DEFAULT_WEIGHTS,
  damageDealt: 0,
  damageTaken: 0,
  kill: 0,
  death: 0,
  exploreMap: 0,
  revisitMap: 0,
  approach: 0,
  onTarget: 0,
  aimedShot: 0,
};

/**
 * The terms that exist to get a policy started, rather than to say what winning
 * is.
 *
 * Aiming pays nothing in this game; covering ground pays nothing; walking
 * toward somebody pays nothing. Each of these is a rung up to something that
 * does pay, and each one is also a way to score without playing well — a worm
 * that keeps a clear line on somebody and never closes is collecting, and so is
 * one that tours the map. Useful early, a distraction later, which is what
 * `shaping` is for: 1 is the whole ladder, 0 is none of it.
 *
 * Damage, kills and deaths are not here. Neither are the stuck and doubling-back
 * penalties: those are not rungs, they are what stops a worm sitting in a hole,
 * and they should still be true at the end.
 */
export const LADDER = ["fromExplore", "fromApproach", "fromOnTarget", "fromAimedShot"];

/** The blank each step's events are read into, one per agent. */
export function emptyEvents(agents) {
  return Array.from({ length: agents }, () => ({
    damageDealt: 0,
    damageTaken: 0,
    selfDamage: 0,
    killed: 0,
    died: 0,
    suicides: 0,
  }));
}

/**
 * Turn the engine's raw hit and kill records into per-agent events.
 *
 * `damage` is (victim, attacker, health) triples and `kills` is (victim, killer)
 * pairs, exactly as `watchDamage` collected them. A worm is often its own
 * attacker — its own grenade, or a fall — and that damage counts against it
 * without counting for it.
 */
export function tallyDamage({ damage, kills }, agents, into = emptyEvents(agents)) {
  for (const events of into) {
    events.damageDealt = 0;
    events.damageTaken = 0;
    events.selfDamage = 0;
    events.killed = 0;
    events.died = 0;
    events.suicides = 0;
  }
  for (let at = 0; at < damage.length; at += 3) {
    const victim = damage[at];
    const attacker = damage[at + 1];
    const amount = damage[at + 2];
    if (victim >= 0 && victim < agents) into[victim].damageTaken += amount;
    if (attacker >= 0 && attacker < agents) {
      if (attacker === victim) into[attacker].selfDamage += amount;
      else into[attacker].damageDealt += amount;
    }
  }
  for (let at = 0; at < kills.length; at += 2) {
    const victim = kills[at];
    const killer = kills[at + 1];
    // Blowing yourself up is a death, not a kill — and it is counted apart, so
    // a death that was nobody's kill can be charged for on its own.
    if (killer >= 0 && killer < agents && killer !== victim) into[killer].killed++;
    else if (victim >= 0 && victim < agents) into[victim].suicides++;
  }
  return into;
}

/**
 * One step's reward, and the terms it is made of. The breakdown is returned
 * because a single number cannot be tuned: when a run goes wrong the question is
 * always which term was doing the talking.
 */
export function combatReward(events, progress, weights = DEFAULT_WEIGHTS, shaping = 1) {
  // Named apart from the events they come from: one is health, the other is
  // points, and a running total that adds both under one name is neither.
  const parts = {
    fromDamageDealt: events.damageDealt * weights.damageDealt,
    fromDamageTaken: -events.damageTaken * weights.damageTaken,
    fromKill: events.killed * weights.kill,
    fromDeath: -events.died * weights.death,
    fromSuicide: -((events.suicides ?? 0) * (weights.suicide ?? 0)) || 0,
    fromExplore: (progress.novel / (progress.cells ?? 1)) * weights.exploreMap,
    fromRevisit: -(progress.revisit / (progress.cells ?? 1)) * weights.revisitMap,
    fromStuck: progress.stuck ? -weights.stuck : 0,
    fromApproach: (progress.approach ?? 0) * weights.approach,
    fromOnTarget: (progress.onTarget ?? 0) * weights.onTarget,
    fromAimedShot: (progress.aimedShot ?? 0) * weights.aimedShot,
    fromGoal:
      progress.goalDelta * weights.goalProgress +
      (progress.reachedGoal ? weights.reachedGoal : 0),
  };
  if (shaping !== 1) {
    for (const name of LADDER) parts[name] *= shaping;
  }
  let reward = 0;
  for (const value of Object.values(parts)) reward += value;
  return { reward, parts, shaping };
}

/** Running totals, for reading a rollout back afterwards. */
export function addEvents(total, events) {
  for (const [name, value] of Object.entries(events)) {
    if (typeof value !== "number") continue;
    total[name] = (total[name] ?? 0) + value;
  }
  return total;
}
