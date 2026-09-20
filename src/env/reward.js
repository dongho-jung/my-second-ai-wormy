// What the agent is paid for.
//
// Damage the other worm loses, minus damage this one loses, plus a kill, minus
// a death. Nothing else — and deliberately nothing that forbids anything. A
// worm that fires a bazooka at its own feet loses health and is paid for it in
// the same currency as being shot, so "do not blow yourself up" is something
// the agent works out rather than a rule someone has to write and tune. In the
// earlier repository self-inflicted damage was 41-56% of all damage taken, so
// there is a great deal for it to work out.
import { nearestFoe } from "./observation.js";

export const DEFAULT_WEIGHTS = {
  // A full 100 health is worth one point either way, so a kill is roughly worth
  // as much as the damage it took to get there.
  damageDealt: 1 / 100,
  damageTaken: 1 / 100,
  kill: 1,
  death: 1,
};

/** The few numbers a reward is computed from, taken before and after a step. */
export function scoreOf(view) {
  const foe = nearestFoe(view) ?? view.foes.find((one) => one.alive) ?? null;
  return {
    alive: Boolean(view.self.alive),
    health: view.self.alive ? view.self.health : 0,
    foeAlive: Boolean(foe?.alive),
    foeHealth: foe?.alive ? foe.health : 0,
  };
}

/**
 * Health lost between two samples of one worm.
 *
 * Death and respawn are the two cases a plain subtraction gets wrong. A worm
 * that died lost whatever it still had, and a worm that respawned came back at
 * full health without having gained anything, so neither is a difference.
 * Picking up a medkit is a gain, not negative damage, and counts as nothing.
 */
export function healthLost(before, after, aliveKey = "alive", healthKey = "health") {
  if (!before[aliveKey]) return 0;
  if (!after[aliveKey]) return before[healthKey];
  return Math.max(0, before[healthKey] - after[healthKey]);
}

/**
 * The reward for one step, and the events behind it — kept separate so a run
 * can be read back as "how much damage, how many kills" and not just a number.
 */
export function combatReward(before, after, weights = DEFAULT_WEIGHTS) {
  const events = {
    damageDealt: healthLost(before, after, "foeAlive", "foeHealth"),
    damageTaken: healthLost(before, after),
    killed: before.foeAlive && !after.foeAlive ? 1 : 0,
    died: before.alive && !after.alive ? 1 : 0,
  };
  const reward =
    events.damageDealt * weights.damageDealt -
    events.damageTaken * weights.damageTaken +
    events.killed * weights.kill -
    events.died * weights.death;
  return { reward, events };
}

/** Running totals, for reading a rollout back afterwards. */
export function addEvents(total, events) {
  for (const [name, value] of Object.entries(events)) {
    total[name] = (total[name] ?? 0) + value;
  }
  return total;
}
