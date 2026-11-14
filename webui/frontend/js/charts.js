// Four live time-series charts (accuracy, stress, regret, maturity),
// fed one point per WebSocket snapshot. Keeps the last MAX_POINTS only
// so a long run doesn't grow the DOM/chart unbounded.
const Charts = (() => {
  const MAX_POINTS = 300;
  let charts = {};
  let lastPushedStage = -1;

  function makeChart(canvasId, label, color, extraDatasets = []) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    return new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          { label, data: [], borderColor: color, tension: 0.2, pointRadius: 0 },
          ...extraDatasets,
        ],
      },
      options: {
        animation: false,
        scales: { x: { display: true, title: { display: true, text: "stage" } } },
        plugins: { legend: { display: extraDatasets.length > 0 }, title: { display: true, text: label, color: "#8b949e" } },
      },
    });
  }

  function init() {
    charts.accuracy = makeChart("chart-accuracy", "val accuracy (this stage)", "#2ea043", [
      {
        label: "best so far (champion)",
        data: [],
        borderColor: "#3fb950",
        borderDash: [6, 4],
        tension: 0,
        pointRadius: 0,
      },
    ]);
    charts.stress = makeChart("chart-stress", "system_stress", "#f85149");
    charts.regret = makeChart("chart-regret", "regret", "#d29922");
    charts.maturity = makeChart("chart-maturity", "maturity", "#58a6ff");
  }

  function push(chart, x, y, y2 = null) {
    chart.data.labels.push(x);
    chart.data.datasets[0].data.push(y);
    if (y2 != null && chart.data.datasets.length > 1) {
      chart.data.datasets[1].data.push(y2);
    }
    if (chart.data.labels.length > MAX_POINTS) {
      chart.data.labels.shift();
      chart.data.datasets.forEach((ds) => ds.data.shift());
    }
    chart.update("none");
  }

  function onSnapshot(s) {
    if (s.stage == null || s.stage < 0) return;
    if (s.stage === lastPushedStage) return;
    lastPushedStage = s.stage;
    push(charts.accuracy, s.stage, s.accuracy, s.best_accuracy);
    push(charts.stress, s.stage, s.system_stress);
    push(charts.regret, s.stage, s.regret);
    push(charts.maturity, s.stage, s.maturity);
  }

  function reset() {
    lastPushedStage = -1;
    for (const c of Object.values(charts)) {
      c.data.labels = [];
      c.data.datasets.forEach((ds) => { ds.data = []; });
      c.update("none");
    }
  }

  return { init, onSnapshot, reset };
})();
