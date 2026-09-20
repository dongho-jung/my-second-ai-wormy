import { createHash } from "node:crypto";
import { CLIENT_SHA256, snapshotV20 } from "./adapter-v20.js";
import { createNullLogger } from "./log.js";

// Reads the live game through the Chrome DevTools Protocol: find the game's own
// controller object inside its closure, then call the adapter on it. The
// exploratory arrays are released immediately and the controller reference is
// dropped the moment the page leaves the room. The whole heap is never queried
// per frame.
export class WebLieroObserver {
  constructor(page, { log = createNullLogger() } = {}) {
    this.page = page;
    this.log = log;
    this.generation = 0;
    this.verified = false;
    this.problem = null;
    this.pending = Promise.resolve();
    this.onNavigation = (frame) => {
      if (frame !== page.mainFrame()) return;
      this.generation++;
      this.verified = false;
      this.problem = null;
      this.controller = null;
      this.prototype = null;
    };
    this.onResponse = (response) => {
      const url = new URL(response.url());
      if (
        url.hostname !== "www.webliero.com" ||
        !url.pathname.endsWith("/game-min.js")
      )
        return;
      const generation = this.generation;
      void response
        .body()
        .then((body) => {
          if (generation !== this.generation) return;
          const digest = createHash("sha256").update(body).digest("hex");
          this.verified =
            url.pathname === "/v/20/game-min.js" && digest === CLIENT_SHA256;
          this.problem = this.verified
            ? null
            : "Unsupported WebLiero client: update the versioned adapter before collecting state.";
          // The single most confusing failure: without this the run sits on
          // "loading" forever and never says the bundle did not match.
          this.log[this.verified ? "info" : "error"]("client_bundle", {
            path: url.pathname,
            sha256: digest,
            verified: this.verified,
          });
        })
        .catch((error) =>
          this.log.warn("client_bundle_unreadable", { message: error.message }),
        );
    };
    page.on("framenavigated", this.onNavigation);
    page.on("response", this.onResponse);
  }

  // Serialize every CDP round trip: state samples, terrain reads and disposal
  // all share one session.
  exclusive(task) {
    const result = this.pending.then(task);
    this.pending = result.catch(() => {});
    return result;
  }

  async call(
    objectId,
    fn,
    args = [],
    returnByValue = false,
    objectGroup = "wormy",
  ) {
    const result = await this.cdp.send("Runtime.callFunctionOn", {
      objectId,
      functionDeclaration: fn.toString(),
      arguments: args.map((value) => ({ value })),
      returnByValue,
      objectGroup,
    });
    if (result.exceptionDetails)
      throw new Error(
        result.exceptionDetails.exception?.description ??
          result.exceptionDetails.text,
      );
    return result.result;
  }

  async discover() {
    const generation = this.generation;
    const requireCurrentPage = () => {
      if (generation !== this.generation)
        throw new Error("Page changed during controller discovery");
    };
    this.cdp ??= await this.page.context().newCDPSession(this.page);
    if (!this.prototype) {
      try {
        const base = await this.cdp.send("Runtime.evaluate", {
          expression: "Function.prototype",
          objectGroup: "wormy-discovery",
        });
        const functions = await this.cdp.send("Runtime.queryObjects", {
          prototypeObjectId: base.result.objectId,
          objectGroup: "wormy-discovery",
        });
        const prototype = await this.call(
          functions.objects.objectId,
          function () {
            const matches = this.filter(
              (fn) =>
                fn.name === "ub" &&
                typeof fn.prototype?.zt === "function" &&
                typeof fn.prototype?.lo === "function",
            );
            if (matches.length !== 1)
              throw new Error(
                "WebLiero controller prototype is ambiguous or unavailable",
              );
            return matches[0].prototype;
          },
        );
        requireCurrentPage();
        this.prototype = prototype.objectId;
      } finally {
        await this.cdp.send("Runtime.releaseObjectGroup", {
          objectGroup: "wormy-discovery",
        });
      }
    }
    try {
      const instances = await this.cdp.send("Runtime.queryObjects", {
        prototypeObjectId: this.prototype,
        objectGroup: "wormy-discovery",
      });
      const controller = await this.call(
        instances.objects.objectId,
        function () {
          const matches = this.filter(
            (value) => value.w?.m?.isConnected && value.Ub?.X?.B instanceof Map,
          );
          if (matches.length > 1)
            throw new Error("More than one active WebLiero controller");
          return matches[0] ?? null;
        },
      );
      requireCurrentPage();
      this.controller = controller.objectId ?? null;
    } finally {
      await this.cdp.send("Runtime.releaseObjectGroup", {
        objectGroup: "wormy-discovery",
      });
    }
  }

  // Attaching to a page that already finished loading means the bundle
  // response is long gone, so it is hashed in place instead. Same file, same
  // check, just fetched from the page's own cache.
  async verifyLoadedBundle() {
    const generation = this.generation;
    if (this.verified || this.problem || this.verifying) return this.verified;
    this.verifying = true;
    try {
      const digest = await this.page.evaluate(async () => {
        if (!/webliero\.com/.test(location.hostname)) return null;
        const response = await fetch("/v/20/game-min.js", {
          cache: "force-cache",
        });
        if (!response.ok) return null;
        const bytes = await response.arrayBuffer();
        const hash = await crypto.subtle.digest("SHA-256", bytes);
        return [...new Uint8Array(hash)]
          .map((byte) => byte.toString(16).padStart(2, "0"))
          .join("");
      });
      if (generation !== this.generation || !digest) return false;
      this.verified = digest === CLIENT_SHA256;
      this.problem = this.verified
        ? null
        : "Unsupported WebLiero client: update the versioned adapter before collecting state.";
      this.log[this.verified ? "info" : "error"]("client_bundle_rehashed", {
        sha256: digest,
        verified: this.verified,
      });
      return this.verified;
    } catch (error) {
      this.log.debug("client_bundle_rehash_failed", { message: error.message });
      return false;
    } finally {
      this.verifying = false;
    }
  }

  read(options = {}) {
    return this.exclusive(async () => {
      if (this.closed || this.page.isClosed())
        return { status: "disconnected", game: null };
      if (!this.verified && !this.problem) await this.verifyLoadedBundle();
      if (this.problem)
        return {
          status: "unsupported_client",
          message: this.problem,
          game: null,
        };
      if (!this.verified) return { status: "loading", game: null };
      const generation = this.generation;
      try {
        const view = await this.page.evaluate(() => {
          const game = Boolean(document.querySelector(".game-view"));
          return {
            game,
            captcha:
              !game &&
              (Boolean(document.querySelector('iframe[src*="recaptcha"]')) ||
                document.body.innerText.includes("Only humans")),
          };
        });
        if (!view.game) {
          if (this.controller)
            await this.cdp.send("Runtime.releaseObject", {
              objectId: this.controller,
            });
          this.controller = null;
          return {
            status: view.captcha ? "captcha_required" : "awaiting_room",
            game: null,
          };
        }
        if (!this.controller) await this.discover();
        if (!this.controller) return { status: "awaiting_room", game: null };
        const result = await this.call(this.controller, snapshotV20, [options], true);
        if (generation !== this.generation)
          return { status: "loading", game: null };
        if (result.value === null) {
          await this.cdp.send("Runtime.releaseObject", {
            objectId: this.controller,
          });
          this.controller = null;
          return { status: "awaiting_room", game: null };
        }
        return { status: "connected", game: result.value };
      } catch (error) {
        if (generation !== this.generation || this.page.isClosed())
          return { status: "loading", game: null };
        // Which phase threw matters: a failed discovery means the adapter no
        // longer matches the page, a failed snapshot means a field mapping does.
        this.log.error("read_failed", {
          phase: this.controller ? "snapshot" : "discovery",
          message: error.message,
          terrain: Boolean(options.terrain || options.terrainPatch),
        });
        return { status: "error", message: error.message, game: null };
      }
    });
  }

  close() {
    this.closed = true;
    this.page.off("framenavigated", this.onNavigation);
    this.page.off("response", this.onResponse);
    return this.exclusive(async () => {
      if (this.cdp) {
        await this.cdp
          .send("Runtime.releaseObjectGroup", { objectGroup: "wormy" })
          .catch(() => {});
        await this.cdp.detach().catch(() => {});
      }
    });
  }
}
