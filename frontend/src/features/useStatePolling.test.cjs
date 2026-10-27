const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const ts = require("typescript");

function harness() {
  const timers = new Map(),
    requests = [],
    renders = [];
  const document = new EventTarget();
  document.hidden = false;
  const window = new EventTarget();
  let cleanup;
  const exports = {};
  const code = ts.transpileModule(
    fs.readFileSync(`${__dirname}/useStatePolling.ts`, "utf8"),
    {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022,
      },
    },
  ).outputText;
  vm.runInNewContext(code, {
    exports,
    document,
    window,
    AbortController,
    require: () => ({
      useRef: (current) => ({ current }),
      useCallback: (fn) => fn,
      useEffect: (fn) => {
        cleanup = fn();
      },
    }),
    setTimeout: (fn, ms) => {
      const id = {};
      timers.set(id, { fn, ms });
      return id;
    },
    clearTimeout: (id) => timers.delete(id),
    fetch: (url, options) =>
      new Promise((resolve) => requests.push({ options, resolve })),
  });
  const refresh = exports.useStatePolling(
    (state) => renders.push(state),
    (message) => assert.fail(message),
  );
  const reply = async (index, status, active = false, tag = '"one"') => {
    requests[index].resolve({
      status,
      ok: status === 200,
      headers: new Map([["ETag", tag]]),
      json: async () => ({ deletion_blocked: active, jobs: [] }),
    });
    await new Promise(setImmediate);
  };
  return {
    timers,
    requests,
    renders,
    document,
    window,
    refresh,
    reply,
    cleanup: () => cleanup(),
  };
}

test("idle uses 30 seconds, 304 does not render, active jobs use 2.5 seconds", async () => {
  const h = harness();
  await h.reply(0, 200);
  assert.equal([...h.timers.values()][0].ms, 30000);
  h.refresh();
  await h.reply(1, 304);
  assert.equal(h.renders.length, 1);
  assert.equal(h.requests[1].options.headers["If-None-Match"], '"one"');
  h.refresh();
  await h.reply(2, 200, true, '"two"');
  assert.equal([...h.timers.values()][0].ms, 2500);
  h.cleanup();
  assert.equal(h.timers.size, 0);
});
test("hidden tabs stop polling and becoming visible refreshes immediately", async () => {
  const h = harness();
  await h.reply(0, 200);
  h.document.hidden = true;
  h.document.dispatchEvent(new Event("visibilitychange"));
  assert.equal(h.timers.size, 0);
  h.document.hidden = false;
  h.document.dispatchEvent(new Event("visibilitychange"));
  assert.equal(h.requests.length, 2);
  await h.reply(1, 304);
  h.cleanup();
});
test("refresh during a request discards the stale result and never overlaps requests", async () => {
  const h = harness();
  h.refresh();
  h.refresh();
  assert.equal(h.requests.length, 1);
  await h.reply(0, 200);
  assert.equal(h.renders.length, 0);
  assert.equal(h.requests.length, 2);
  await h.reply(1, 200, true);
  assert.equal(h.renders.length, 1);
  h.cleanup();
  assert.equal(h.requests[1].options.signal.aborted, true);
});
