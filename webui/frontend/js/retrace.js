// Stage-scrubber playback of a completed run's journal entries.
const Retrace = (() => {
  let entries = [];

  function init() {
    document.getElementById("btn-load-retrace").onclick = load;
    document.getElementById("retrace-scrubber").oninput = (e) => show(parseInt(e.target.value, 10));
  }

  async function load() {
    const dataset = document.getElementById("retrace-dataset").value.trim();
    const foldStr = document.getElementById("retrace-fold").value;
    const fold = foldStr === "" ? null : parseInt(foldStr, 10);
    if (!dataset) return;
    const q = fold !== null ? `?fold=${fold}` : "";
    const data = await Api.get(`/retrace/${encodeURIComponent(dataset)}${q}`);
    entries = data.entries || [];
    const scrubber = document.getElementById("retrace-scrubber");
    scrubber.max = Math.max(0, entries.length - 1);
    scrubber.value = 0;
    show(0);
  }

  function show(idx) {
    if (!entries.length) {
      document.getElementById("retrace-detail").textContent = "(no entries)";
      return;
    }
    const e = entries[idx];
    document.getElementById("retrace-position").textContent = `stage ${idx + 1} / ${entries.length}`;
    document.getElementById("retrace-detail").textContent = JSON.stringify(e, null, 2);
  }

  return { init };
})();
