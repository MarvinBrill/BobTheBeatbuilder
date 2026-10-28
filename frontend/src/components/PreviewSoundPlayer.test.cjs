const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const ts = require("typescript");

function harness(events, missing = false) {
  const timers = new Set(),
    sources = [],
    levels = [],
    status = [];
  const audio = new EventTarget();
  Object.assign(audio, {
    currentTime: 0,
    paused: true,
    seeking: false,
    readyState: 4,
    playbackRate: 1,
  });
  let context;
  class Context {
    currentTime = 10;
    destination = {};
    constructor() {
      context = this;
    }
    async resume() {}
    async close() {
      this.closed = true;
    }
    async decodeAudioData() {
      return { duration: 0.5 };
    }
    createGain() {
      const node = {
        gain: {
          value: 1,
          setTargetAtTime(value) {
            this.value = value;
          },
        },
        connect() {
          return this;
        },
        disconnect() {},
      };
      levels.push(node);
      return node;
    }
    createBufferSource() {
      const source = {
        playbackRate: {},
        connect(node) {
          return node;
        },
        disconnect() {},
        start(...args) {
          this.started = args;
        },
        stop(when) {
          if (when === undefined) this.cancelled = true;
          else this.stopAt = when;
        },
      };
      sources.push(source);
      return source;
    }
  }
  const exports = {};
  vm.runInNewContext(
    ts.transpileModule(
      fs.readFileSync(`${__dirname}/PreviewSoundPlayer.ts`, "utf8"),
      {
        compilerOptions: {
          module: ts.ModuleKind.CommonJS,
          target: ts.ScriptTarget.ES2022,
        },
      },
    ).outputText,
    {
      exports,
      AudioContext: Context,
      AbortController,
      setInterval(fn) {
        timers.add(fn);
        return fn;
      },
      clearInterval(fn) {
        timers.delete(fn);
      },
      fetch: async (url) => ({
        ok: true,
        arrayBuffer: async () => new ArrayBuffer(1),
        json: async () => ({
          name: missing ? null : "ssM4in",
          samples: missing
            ? {}
            : {
                hitnormal: "/hit",
                slidertick: "/tick",
                sliderslide: "/slide",
              },
          warning: missing ? "Skin fehlt" : null,
        }),
      }),
    },
  );
  const player = new exports.PreviewSoundPlayer(audio, events, (text) =>
    status.push(text),
  );
  return {
    player,
    audio,
    sources,
    levels,
    timers,
    status,
    get context() {
      return context;
    },
    fire: (name) => audio.dispatchEvent(new Event(name)),
    step(position) {
      context.currentTime += position - audio.currentTime;
      audio.currentTime = position;
      [...timers].forEach((fn) => fn());
    },
  };
}

const hit = (time) => ({ time, sample: "hitnormal", volume: 0.7 });
const loop = { time: 1, end: 3, sample: "sliderslide", volume: 0.7 };

test("sample timing uses music clock and separate volume; pause cancels all audio", async () => {
  const h = harness([hit(1), loop, hit(3)]);
  assert.equal(h.timers.size, 0);
  await h.player.prepare();
  h.player.setVolume(0.25);
  h.audio.paused = false;
  h.fire("playing");
  h.step(0.9);
  assert.equal(h.sources.length, 2);
  assert.ok(Math.abs(h.sources[0].started[0] - 11) < 1e-9);
  assert.equal(h.sources[1].loop, true);
  assert.equal(h.sources[1].stopAt, 13);
  assert.equal(h.levels[0].gain.value, 0.25);
  assert.equal(h.levels[1].gain.value, 0.7);
  h.audio.paused = true;
  h.fire("pause");
  assert.equal(h.timers.size, 0);
  assert.ok(h.sources.every((source) => source.cancelled));
  h.player.dispose();
  assert.equal(h.context.closed, true);
});

test("seek into slider resumes loop without replaying hits; buffering cancels playback", async () => {
  const h = harness([hit(1), loop, hit(1.5), hit(3)]);
  await h.player.prepare();
  h.audio.paused = false;
  h.audio.seeking = true;
  h.fire("seeking");
  h.audio.currentTime = 2;
  h.audio.seeking = false;
  h.fire("seeked");
  assert.equal(h.sources.length, 1);
  assert.equal(h.sources[0].loop, true);
  assert.equal(h.sources[0].stopAt, 11);
  h.fire("waiting");
  assert.equal(h.timers.size, 0);
  assert.equal(h.sources[0].cancelled, true);
  h.player.dispose();
});

test("delayed scheduler skips missed hits; map switch releases listeners and timers", async () => {
  const h = harness([hit(1), hit(1.2), hit(1.4), hit(2.05)]);
  await h.player.prepare();
  h.audio.paused = false;
  h.fire("playing");
  h.step(2);
  assert.equal(h.sources.length, 1);
  h.player.dispose();
  h.fire("playing");
  assert.equal(h.timers.size, 0);
  assert.equal(h.sources.length, 1);
});

test("missing skin leaves playback usable with an explanatory status", async () => {
  const h = harness([hit(1)], true);
  await h.player.prepare();
  h.audio.paused = false;
  h.fire("playing");
  h.step(0.95);
  assert.equal(h.sources.length, 0);
  assert.match(h.status.at(-1), /Skin fehlt/);
  h.player.dispose();
});
