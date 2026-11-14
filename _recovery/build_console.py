from pathlib import Path

root = Path(r"c:\Users\arinb\Downloads\figpso_uil")
rec = root / "_recovery"

kg_section = """
      <section class="view" id="view-kg">
        <div class="kg-workspace">
          <div class="kg-float-bar">
            <div class="kg-title-block">
              <span class="kg-eyebrow">Neural knowledge layer</span>
              <h3>Editable Network · synapse map</h3>
              <span class="muted" id="kgFocusLabel"></span>
            </div>
            <div class="seg glass" id="kgModeSeg">
              <button data-v="network" class="on">Network</button>
              <button data-v="brain">Mindmap</button>
              <button data-v="orbit">Orbit</button>
            </div>
            <div class="chips compact" id="kgEdgeChips">
              <button class="chip on" data-t="">all</button>
              <button class="chip" data-t="leads_to">leads_to</button>
              <button class="chip" data-t="strengthens">strengthens</button>
              <button class="chip" data-t="contradicts">contradicts</button>
              <button class="chip" data-t="depends_on">depends_on</button>
            </div>
            <input class="kg-search glass" id="kgSearch" placeholder="Search neurons...">
            <div class="kg-toggles">
              <label class="toggle"><input type="checkbox" id="kgLabels" checked><span>labels</span></label>
              <label class="toggle"><input type="checkbox" id="kgPhysics" checked><span>physics</span></label>
              <label class="toggle"><input type="checkbox" id="kgPulse" checked><span>pulse</span></label>
              <label class="toggle"><input type="checkbox" id="kgDetail" checked><span>detail</span></label>
            </div>
            <button class="btn ghost sm" id="kgFit">Fit</button>
            <button class="btn ghost sm hidden" id="kgClearFocus">Clear focus</button>
            <button class="btn ghost sm" id="kgRefresh">Refresh</button>
          </div>
          <div class="kg-layout">
            <div class="card kg-canvas-card">
              <canvas id="kgCanvas"></canvas>
              <div class="kg-hovercard hidden" id="kgHover"></div>
              <div class="kg-legend" id="kgLegend"></div>
            </div>
            <div class="kg-side">
              <div class="card"><div class="card-title">Vitals</div><div id="kgVitals"></div></div>
              <div class="card"><div class="card-title">Edge types</div><div id="kgTypeBars"></div></div>
              <div class="card"><div class="card-title">Top nodes</div><div id="kgTopNodes"></div></div>
              <div class="card" id="kgNodeEditor"><div class="empty small"><p>Select a node</p></div></div>
              <div class="card" id="kgEdgeEditor"><div class="empty small"><p>Select an edge</p></div></div>
              <div class="hidden">
                <input id="kgConfSlider" type="range" min="0" max="100" value="50">
                <span id="kgConfLive">0.50</span>
                <select id="kgTierSelect"><option value="0">0</option><option value="1">1</option><option value="2">2</option><option value="3">3</option></select>
                <button id="kgSaveNode">Save</button>
                <button id="kgFocusNode">Focus</button>
                <button id="kgAdvEdit">Advanced</button>
                <button id="kgJumpSource">Source</button>
                <button id="kgJumpTarget">Target</button>
              </div>
            </div>
          </div>
        </div>
      </section>
"""

# Load HTML template parts from companion files if present; else build inline minimal via exec of parts file
parts = root / "_recovery" / "build_console_parts.py"
print("run companion")
