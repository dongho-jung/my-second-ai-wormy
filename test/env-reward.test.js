import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_WEIGHTS,
  addEvents,
  combatReward,
  healthLost,
  scoreOf,
} from "../src/env/reward.js";

const score = (fields) => ({
  alive: true,
  health: 100,
  foeAlive: true,
  foeHealth: 100,
  ...fields,
});

test("damage dealt pays and damage taken costs, at the same rate", () => {
  const { reward, events } = combatReward(
    score(),
    score({ health: 80, foeHealth: 65 }),
  );
  assert.deepEqual(events, {
    damageDealt: 35,
    damageTaken: 20,
    killed: 0,
    died: 0,
  });
  // 0.35 - 0.20 in binary floating point, so the comparison is rounded.
  assert.equal(+reward.toFixed(10), 0.15);
});

test("a worm that blows itself up is charged for it, with no special case", () => {
  // The only thing that happened is 40 health gone from this worm, which is
  // what a bazooka at your own feet looks like from here.
  const { reward, events } = combatReward(score(), score({ health: 60 }));
  assert.equal(events.damageTaken, 40);
  assert.equal(reward, -0.4);
});

test("a kill counts the lethal blow and the kill; a death counts both too", () => {
  const kill = combatReward(
    score({ foeHealth: 12 }),
    score({ foeAlive: false, foeHealth: 0 }),
  );
  assert.deepEqual(kill.events, {
    damageDealt: 12,
    damageTaken: 0,
    killed: 1,
    died: 0,
  });
  assert.equal(kill.reward, 1.12);
  const death = combatReward(
    score({ health: 30 }),
    score({ alive: false, health: 0 }),
  );
  assert.equal(death.events.died, 1);
  assert.equal(death.reward, -1.3);
});

test("a respawn is not free healing and a medkit is not negative damage", () => {
  const respawn = combatReward(
    score({ alive: false, health: 0, foeAlive: false, foeHealth: 0 }),
    score(),
  );
  assert.deepEqual(respawn.events, {
    damageDealt: 0,
    damageTaken: 0,
    killed: 0,
    died: 0,
  });
  assert.equal(respawn.reward, 0);
  const medkit = combatReward(score({ health: 40 }), score({ health: 90 }));
  assert.equal(medkit.events.damageTaken, 0, "a gain is not damage");
  assert.equal(medkit.reward, 0);
});

test("health lost is read off the alive transition, not the subtraction", () => {
  assert.equal(healthLost({ alive: true, health: 70 }, { alive: true, health: 55 }), 15);
  assert.equal(
    healthLost({ alive: true, health: 70 }, { alive: false, health: 0 }),
    70,
    "dying loses whatever was left",
  );
  assert.equal(
    healthLost({ alive: false, health: 0 }, { alive: true, health: 100 }),
    0,
    "a worm that was already dead cannot lose health",
  );
});

test("weights are the one place the balance lives", () => {
  const brutal = { ...DEFAULT_WEIGHTS, kill: 10, death: 0 };
  const { reward } = combatReward(
    score({ foeHealth: 5 }),
    score({ health: 0, alive: false, foeAlive: false, foeHealth: 0 }),
    brutal,
  );
  assert.equal(reward, 0.05 - 1 + 10, "a trade is worth taking at kill 10");
});

test("a score is read from a view, and takes the nearest living foe", () => {
  const view = {
    self: { alive: true, health: 55, position: { x: 0, y: 0 } },
    foes: [
      { alive: false, position: { x: 1, y: 0 }, health: 0 },
      { alive: true, position: { x: 40, y: 0 }, health: 30 },
      { alive: true, position: { x: 10, y: 0 }, health: 80 },
    ],
    projectiles: [],
  };
  assert.deepEqual(scoreOf(view), {
    alive: true,
    health: 55,
    foeAlive: true,
    foeHealth: 80,
  });
  assert.deepEqual(scoreOf({ self: { alive: false }, foes: [], projectiles: [] }), {
    alive: false,
    health: 0,
    foeAlive: false,
    foeHealth: 0,
  });
});

test("event totals add up across a rollout", () => {
  const total = {};
  addEvents(total, { damageDealt: 3, damageTaken: 0, killed: 0, died: 0 });
  addEvents(total, { damageDealt: 4, damageTaken: 9, killed: 1, died: 0 });
  assert.deepEqual(total, { damageDealt: 7, damageTaken: 9, killed: 1, died: 0 });
});
