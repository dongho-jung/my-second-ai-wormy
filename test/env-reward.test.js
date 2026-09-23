import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_WEIGHTS, LADDER, addEvents, combatReward, emptyEvents, tallyDamage } from "../src/env/reward.js";
import { Progress } from "../src/env/progress.js";

const still = {
  novel: 0,
  revisit: 0,
  stuck: false,
  stuckSteps: 0,
  goalDelta: 0,
  reachedGoal: false,
};

test("a hit is credited to whoever landed it, in a crowd", () => {
  // 0 hits 1 for 30; 2 hits 1 for 10 and finishes it; 0 hurts itself for 20.
  const events = tallyDamage(
    { damage: [1, 0, 30, 1, 2, 10, 0, 0, 20], kills: [1, 2] },
    3,
  );
  assert.deepEqual(events[0], {
    damageDealt: 30,
    damageTaken: 20,
    selfDamage: 20,
    killed: 0,
    died: 0,
    suicides: 0,
  });
  assert.deepEqual(events[1], {
    damageDealt: 0,
    damageTaken: 40,
    selfDamage: 0,
    killed: 0,
    died: 0,
    suicides: 0,
  });
  assert.equal(events[2].damageDealt, 10);
  assert.equal(events[2].killed, 1, "the kill goes to the one who landed the last hit");
});

test("blowing yourself up is a death and nobody's kill", () => {
  const events = tallyDamage({ damage: [0, 0, 100], kills: [0, 0] }, 2);
  assert.equal(events[0].damageTaken, 100);
  assert.equal(events[0].selfDamage, 100);
  assert.equal(events[0].damageDealt, 0, "hurting yourself is not dealing damage");
  assert.equal(events[0].killed, 0);
  assert.equal(events[1].killed, 0, "and it is certainly not the other one's kill");
  assert.equal(events[0].suicides, 1, "but it is counted, so it can be charged for");
  const fell = tallyDamage({ damage: [], kills: [1, -1] }, 2);
  assert.equal(fell[1].suicides, 1, "a death nobody caused is the same kind of death");
});

test("hits from outside the agent list are taken but credited to nobody", () => {
  // A worm the environment is not driving, or the engine's own -1 for no owner.
  const events = tallyDamage({ damage: [0, 7, 25, 0, -1, 5], kills: [] }, 2);
  assert.equal(events[0].damageTaken, 30);
  assert.equal(events[1].damageDealt, 0);
});

test("the buffers are reused, so a step never sees the last one's hits", () => {
  const into = emptyEvents(2);
  tallyDamage({ damage: [1, 0, 30], kills: [] }, 2, into);
  tallyDamage({ damage: [], kills: [] }, 2, into);
  assert.deepEqual(into[0], {
    damageDealt: 0,
    damageTaken: 0,
    selfDamage: 0,
    killed: 0,
    died: 0,
    suicides: 0,
  });
});

test("a death by its own hand can be charged on top, and is not by default", () => {
  const events = { damageDealt: 0, damageTaken: 100, selfDamage: 100, killed: 0, died: 1, suicides: 1 };
  const nothing = combatReward(events, { ...still, cells: 100 });
  assert.equal(nothing.parts.fromSuicide, 0, "off unless a run asks for it");
  const charged = combatReward(events, { ...still, cells: 100 }, { ...DEFAULT_WEIGHTS, suicide: 3 });
  assert.equal(charged.parts.fromSuicide, -3);
  assert.equal(charged.parts.fromDeath, -DEFAULT_WEIGHTS.death, "on top of the death, not instead of it");
  assert.equal(charged.reward, nothing.reward - 3);
});

test("damage pays, being hurt costs, and a kill is worth more than either", () => {
  const { reward, parts } = combatReward(
    { damageDealt: 40, damageTaken: 20, selfDamage: 0, killed: 0, died: 0 },
    still,
  );
  // Binary floating point: 40/100 is not exactly 0.4, so the comparisons round.
  assert.equal(+parts.fromDamageDealt.toFixed(10), 0.4);
  assert.equal(+parts.fromDamageTaken.toFixed(10), -0.1);
  assert.equal(+reward.toFixed(10), 0.3);
  const kill = combatReward(
    { damageDealt: 12, damageTaken: 0, selfDamage: 0, killed: 1, died: 0 },
    still,
  );
  assert.equal(kill.parts.fromKill, 4);
  assert.equal(+kill.reward.toFixed(10), 4.12);
  const death = combatReward(
    { damageDealt: 0, damageTaken: 30, selfDamage: 30, killed: 0, died: 1 },
    still,
  );
  assert.equal(death.parts.fromDeath, -2);
  assert.equal(+death.reward.toFixed(10), -2.15, "its own grenade is charged like any hit");
});

test("an even fight has to pay, or standing still wins", () => {
  // Everyone is the same policy, so over many episodes a worm deals as much as
  // it takes. If that came to zero the best strategy would be to never fire.
  const even = combatReward(
    { damageDealt: 100, damageTaken: 100, selfDamage: 0, killed: 1, died: 1 },
    still,
  );
  assert.ok(
    even.reward > 0.5,
    `trading evenly has to be worth more than doing nothing, got ${even.reward}`,
  );
  // And a worm that only ever hurts itself still loses.
  const clumsy = combatReward(
    { damageDealt: 0, damageTaken: 100, selfDamage: 100, killed: 0, died: 1 },
    still,
  );
  assert.ok(clumsy.reward < -1, `got ${clumsy.reward}`);
});

test("standing in a hole costs, and never enough to make dying the way out", () => {
  const stuck = combatReward(emptyEvents(1)[0], { ...still, stuck: true, stuckSteps: 40 });
  assert.equal(stuck.reward, -DEFAULT_WEIGHTS.stuck);
  // The whole of a 900-step episode spent stuck has to stay under a few deaths,
  // or the shortest path to a better score is suicide.
  const worstCase = 900 * DEFAULT_WEIGHTS.stuck;
  assert.ok(
    worstCase > DEFAULT_WEIGHTS.death,
    "being stuck for a whole episode should still be worse than one death",
  );
  assert.ok(
    worstCase < 6 * DEFAULT_WEIGHTS.death,
    `an episode stuck costs ${worstCase}, which must stay within a few deaths`,
  );
});

test("new ground pays and doubling back costs", () => {
  assert.equal(
    combatReward(emptyEvents(1)[0], { ...still, novel: 1, cells: 300 }).parts.fromExplore,
    3 / 300,
  );
  // Paid as a share of the map: one cell of a 300-cell map is a three-hundredth
  // of what covering all of it is worth.
  assert.equal(
    combatReward(emptyEvents(1)[0], { ...still, revisit: 1, cells: 300 }).parts.fromRevisit,
    -3 / 300,
  );
});

test("a goal pays for closing on it and once for arriving", () => {
  const closing = combatReward(emptyEvents(1)[0], { ...still, goalDelta: 12 });
  assert.equal(+closing.parts.fromGoal.toFixed(10), 0.24);
  const arriving = combatReward(emptyEvents(1)[0], {
    ...still,
    goalDelta: 3,
    reachedGoal: true,
  });
  assert.equal(+arriving.parts.fromGoal.toFixed(10), 2.06);
  const away = combatReward(emptyEvents(1)[0], { ...still, goalDelta: -12 });
  assert.equal(+away.parts.fromGoal.toFixed(10), -0.24, "walking away costs the same");
});

test("a speed fine-tune pays a bounded arrival bonus and can fade distance shaping", () => {
  const weights = {
    ...DEFAULT_WEIGHTS,
    goalSpeed: 2,
    goalSpeedCap: 5,
  };
  const fast = combatReward(
    emptyEvents(1)[0],
    { ...still, goalDelta: 12, reachedGoal: true, goalSpeed: 9 },
    weights,
    1,
    0.25,
  );
  assert.equal(+fast.parts.fromGoal.toFixed(10), 2.06, "only the per-pixel part fades");
  assert.equal(fast.parts.fromGoalSpeed, 10, "speed is capped before its weight is applied");
  assert.equal(+fast.reward.toFixed(10), 12.06);
  const unfinished = combatReward(
    emptyEvents(1)[0],
    { ...still, goalSpeed: 9 },
    weights,
  );
  assert.equal(unfinished.parts.fromGoalSpeed, 0, "moving fast without arriving earns nothing");
});

test("a rope throw costs what the weights say, and nothing by default", () => {
  const thrown = { ...still, ropeThrows: 1 };
  assert.equal(combatReward(emptyEvents(1)[0], thrown).parts.fromRopeThrow, 0, "free by default");
  const charged = combatReward(emptyEvents(1)[0], thrown, { ...DEFAULT_WEIGHTS, ropeThrow: 0.01 });
  assert.equal(+charged.parts.fromRopeThrow.toFixed(10), -0.01);
  assert.equal(+charged.reward.toFixed(10), -0.01, "and it is part of the reward");
  const held = combatReward(emptyEvents(1)[0], still, { ...DEFAULT_WEIGHTS, ropeThrow: 0.01 });
  assert.equal(held.parts.fromRopeThrow, 0, "holding on is free");
});

test("a goal's age is counted, so a caller can give up on it", () => {
  const progress = new Progress({ window: 5 });
  progress.reset({ goal: { x: 900, y: 100 } });
  let facts;
  for (let step = 0; step < 3; step++) facts = progress.update({ x: 100, y: 100 });
  assert.equal(facts.goalSteps, 3);
  progress.setGoal({ x: 950, y: 100 });
  assert.equal(progress.update({ x: 100, y: 100 }).goalSteps, 1, "a new goal starts over");
  progress.setGoal(null);
  assert.equal(progress.update({ x: 100, y: 100 }).goalSteps, 0, "no goal, no age");
});

test("best-distance progress permits a detour without paying twice for the return", () => {
  const progress = new Progress({ teleportPx: 60 });
  progress.setGoal({ x: 200, y: 100 }, { x: 100, y: 100 });
  progress.update({ x: 100, y: 100 });

  const away = progress.update({ x: 80, y: 100 });
  assert.equal(away.goalDelta, -20, "signed progress still describes moving away");
  assert.equal(away.goalBestDelta, 0, "a required retreat is neutral in best mode");

  const back = progress.update({ x: 100, y: 100 });
  assert.equal(back.goalBestDelta, 0, "returning over already credited ground pays nothing");

  const closer = progress.update({ x: 120, y: 100 });
  assert.equal(closer.goalBestDelta, 20, "only a new closest point pays");

  progress.restart();
  const respawned = progress.update({ x: 180, y: 100 });
  assert.equal(respawned.goalBestDelta, 0, "a closer respawn is not movement progress");
});

test("weights are the one place the balance lives", () => {
  const brutal = { ...DEFAULT_WEIGHTS, kill: 10, death: 0 };
  const { reward } = combatReward(
    { damageDealt: 5, damageTaken: 100, selfDamage: 0, killed: 1, died: 1 },
    still,
    brutal,
  );
  assert.equal(+reward.toFixed(10), 0.05 - 0.5 + 10, "a trade is worth taking at kill 10");
});

test("standing still reads as stuck; walking away does not", () => {
  const progress = new Progress({ window: 10, stuckPx: 5 });
  progress.reset();
  let facts;
  for (let step = 0; step < 15; step++) facts = progress.update({ x: 100, y: 100 });
  assert.equal(facts.stuck, true);
  assert.equal(facts.stuckSteps, 5, "counted from the moment the window was full");
  for (let step = 0; step < 15; step++) {
    facts = progress.update({ x: 100 + step * 20, y: 100 });
  }
  assert.equal(facts.stuck, false);
  assert.equal(Math.round(facts.movedPx), 200);
});

test("pacing a short loop is seen, even though it is moving the whole time", () => {
  const progress = new Progress({ window: 10, stuckPx: 5, cellPx: 16, revisitMemory: 40 });
  progress.reset();
  let revisits = 0;
  let novel = 0;
  for (let step = 0; step < 60; step++) {
    const facts = progress.update({ x: 100 + (step % 4) * 20, y: 100 });
    revisits += facts.revisit;
    novel += facts.novel;
  }
  assert.equal(novel, 4, "four cells, seen once each");
  assert.ok(revisits > 50, `and walked over again and again, got ${revisits}`);
  // Crossing new ground the same distance costs nothing and pays four times.
  progress.reset();
  let fresh = 0;
  for (let step = 0; step < 60; step++) fresh += progress.update({ x: 100 + step * 20, y: 100 }).novel;
  assert.equal(fresh, 60);
});

test("a dead or respawned worm is not stuck, it is somewhere else", () => {
  const progress = new Progress({ window: 10, stuckPx: 5, teleportPx: 60 });
  progress.reset();
  for (let step = 0; step < 15; step++) progress.update({ x: 100, y: 100 });
  assert.equal(progress.update({ x: 100, y: 100 }, false).stuck, false, "dead is not stuck");
  // Back at full health on the other side of the map: the trail says nothing.
  const after = progress.update({ x: 400, y: 200 });
  assert.equal(after.stuck, false);
  assert.equal(after.movedPx, null, "nothing to measure against yet");
});

test("the goal is paid for once, however long it sits on the spot", () => {
  const progress = new Progress({ window: 5, goalRadiusPx: 20 });
  progress.reset({ goal: { x: 200, y: 100 } });
  let arrivals = 0;
  let gained = 0;
  for (let step = 0; step < 20; step++) {
    const facts = progress.update({ x: Math.min(200, 100 + step * 20), y: 100 });
    arrivals += facts.reachedGoal ? 1 : 0;
    gained += facts.goalDelta;
  }
  assert.equal(arrivals, 1);
  assert.equal(Math.round(gained), 80, "measured from the second step, when there is a delta");
});

test("a completed goal reports elapsed time, direct distance, route length and speed", () => {
  const progress = new Progress({ window: 5, goalRadiusPx: 20, teleportPx: 60 });
  progress.setGoal({ x: 200, y: 100 }, { x: 100, y: 100 });
  let facts;
  for (const x of [120, 140, 160, 180]) facts = progress.update({ x, y: 100 });
  assert.equal(facts.reachedGoal, true);
  assert.equal(facts.goalSteps, 4);
  assert.equal(facts.goalDirectPx, 80, "the arrival circle is not counted as travel owed");
  assert.equal(facts.goalPathPx, 80);
  assert.equal(facts.goalSpeed, 20);
});

test("a speed phase counts death on the goal clock and never as travel", () => {
  const progress = new Progress({
    window: 5,
    goalRadiusPx: 20,
    teleportPx: 60,
    goalClockWhileDead: true,
  });
  progress.setGoal({ x: 800, y: 100 }, { x: 100, y: 100 });
  progress.update({ x: 120, y: 100 });
  const dead = progress.update({ x: 120, y: 100 }, false);
  const respawned = progress.update({ x: 500, y: 100 });
  assert.equal(dead.goalSteps, 2, "waiting to respawn is not free time");
  assert.equal(respawned.goalSteps, 3);
  assert.equal(progress.goalPathPx, 20, "the jump across the map was not a fast route");

  const compatible = new Progress({ goalRadiusPx: 20 });
  compatible.setGoal({ x: 800, y: 100 }, { x: 100, y: 100 });
  compatible.update({ x: 100, y: 100 }, false);
  assert.equal(compatible.goalSteps, 0, "old runs keep their paused respawn clock by default");
});

test("the ladder can be faded without touching what it leads to", () => {
  const events = { damageDealt: 100, damageTaken: 0, killed: 1, died: 0 };
  const progress = {
    cells: 100, novel: 1, revisit: 0, stuck: true, stuckSteps: 1,
    approach: 10, onTarget: 1, aimedShot: 1, goalDelta: 0, reachedGoal: false,
  };
  const full = combatReward(events, progress, DEFAULT_WEIGHTS, 1);
  const none = combatReward(events, progress, DEFAULT_WEIGHTS, 0);

  // The rungs go to nothing.
  for (const name of LADDER) {
    assert.notEqual(full.parts[name], 0, `${name} should pay something at full`);
    assert.equal(none.parts[name], 0, `${name} should pay nothing at zero`);
  }
  // What they were rungs up to does not move, and neither does the penalty for
  // sitting in a hole: that is not a rung, it is what stops a worm doing it.
  for (const name of ["fromDamageDealt", "fromDamageTaken", "fromKill", "fromDeath", "fromStuck"]) {
    assert.equal(none.parts[name], full.parts[name], `${name} is not a rung`);
  }
  // Half way down is half way down.
  const half = combatReward(events, progress, DEFAULT_WEIGHTS, 0.5);
  assert.equal(half.parts.fromOnTarget, full.parts.fromOnTarget / 2);
  assert.equal(half.reward, (full.reward + none.reward) / 2);
});

test("event totals add up across a rollout", () => {
  const total = {};
  addEvents(total, { damageDealt: 3, damageTaken: 0, killed: 0, died: 0 });
  addEvents(total, { damageDealt: 4, damageTaken: 9, killed: 1, died: 0 });
  assert.deepEqual(total, { damageDealt: 7, damageTaken: 9, killed: 1, died: 0 });
});
