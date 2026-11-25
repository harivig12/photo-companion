// Loaded in <head> before first paint: apply a stored theme choice, otherwise follow the device.
(function () {
  try {
    var t = localStorage.getItem("photos-theme");
    if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  } catch (e) { /* storage blocked: follow the device */ }
})();
