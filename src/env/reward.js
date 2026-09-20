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
  // Ground it has not covered before, per grid cell.
  explore: 0.02,
  // Coming back to somewhere it was recently. Pacing a short loop trips this
  // every step and nothing else notices it.
  revisit: 0.02,
  // Per step spent unable to move. It has to hurt enough to be worth digging
  // out of, and not so much that dying becomes the cheaper way out of a hole.
  // At this weight, being stuck long enough to be worth dying over is about 200
  // steps — thirteen seconds — by which point the worm really has dropped out
  // of the match.
  stuck: 0.01,
  // Per pixel of progress toward a goal, when one is set, and for arriving.
  goalProgress: 0.02,
  reachedGoal: 2,
};

/** The blank each step's events are read into, one per agent. */
export function emptyEvents(agents) {
  return Array.from({ length: agents }, () => ({
    damageDealt: 0,
    damageTaken: 0,
    selfDamage: 0,
    killed: 0,
    died: 0,
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
    // Blowing yourself up is a death, not a kill.
    if (killer >= 0 && killer < agents && killer !== victim) into[killer].killed++;
  }
  return into;
}

/**
 * One step's reward, and the terms it is made of. The breakdown is returned
 * because a single number cannot be tuned: when a run goes wrong the question is
 * always which term was doing the talking.
 */
export function combatReward(events, progress, weights = DEFAULT_WEIGHTS) {
  // Named apart from the events they come from: one is health, the other is
  // points, and a running total that adds both under one name is neither.
  const parts = {
    fromDamageDealt: events.damageDealt * weights.damageDealt,
    fromDamageTaken: -events.damageTaken * weights.damageTaken,
    fromKill: events.killed * weights.kill,
    fromDeath: -events.died * weights.death,
    fromExplore: progress.novel * weights.explore,
    fromRevisit: -progress.revisit * weights.revisit,
    fromStuck: progress.stuck ? -weights.stuck : 0,
    fromGoal:
      progress.goalDelta * weights.goalProgress +
      (progress.reachedGoal ? weights.reachedGoal : 0),
  };
  let reward = 0;
  for (const value of Object.values(parts)) reward += value;
  return { reward, parts };
}

/** Running totals, for reading a rollout back afterwards. */
export function addEvents(total, events) {
  for (const [name, value] of Object.entries(events)) {
    if (typeof value !== "number") continue;
    total[name] = (total[name] ?? 0) + value;
  }
  return total;
}
