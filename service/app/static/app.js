// Theme toggle and live refresh of the running job on the Jobs page. No dependencies.
(function () {
  var btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.addEventListener("click", function () {
      var root = document.documentElement;
      var current = root.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
      var next = current === "light" ? "dark" : "light";
      root.dataset.theme = next;
      try { localStorage.setItem("photos-theme", next); } catch (e) { /* ignore */ }
    });
  }

  var live = document.getElementById("live");
  if (!live) return;
  var runningId = live.dataset.jobId || "";
  var queuedCount = live.dataset.queued || "0";

  function tick() {
    fetch("/api/jobs", { credentials: "same-origin", cache: "no-store" })
      .then(function (r) { if (r.status === 401) { location.href = "/login"; } return r.json(); })
      .then(function (data) {
        var r = data.running;
        var id = r ? String(r.id) : "";
        // A job started or finished, or the queue changed: reload for the full picture.
        if (id !== runningId || String(data.queued.length) !== queuedCount) { location.reload(); return; }
        if (!r) return;
        var bar = document.getElementById("live-progress");
        var stage = document.getElementById("live-stage");
        var log = document.getElementById("live-log");
        if (bar) { bar.value = r.progress; bar.textContent = r.progress + "%"; }
        if (stage) stage.textContent = r.stage + " · " + r.progress + "%";
        if (log) log.textContent = r.log.join("\n");
      })
      .catch(function () { /* offline: keep the last data on screen */ });
  }
  setInterval(tick, 3000);
})();
