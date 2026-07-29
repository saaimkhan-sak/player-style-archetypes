const DATA_ROOT = "/data";

const appState = {
  core: null,
  route: "overview",
  season: null,
  group: "forwards",
  glossaryGroup: "forwards",
  glossaryQuery: "",
  seasonTab: "snapshot",
  selectedPlayerId: null,
  seasonCache: new Map(),
  careers: null,
  careerGroup: "forwards",
  careerQuery: "",
  careerPlayerId: null,
  playoffs: null,
  playoffSeason: null,
  playoffGroup: "forwards",
  playoffTab: "shifts",
  playoffMinReg: 20,
  playoffMinPo: 4,
  playoffQuery: "",
  playoffPlayerId: null,
  canvasCleanups: [],
};

const ROUTE_LABELS = {
  overview: "Overview",
  glossary: "Style glossary",
  season: "Season lab",
  career: "Career paths",
  playoffs: "Playoff pressure",
};

const PROFILE_COLORS = [
  "#21b6a8",
  "#f06445",
  "#5f8dd3",
  "#d7a843",
  "#8c78d4",
  "#3e8c67",
  "#c66c8e",
  "#6a8a94",
];

const main = document.querySelector("#main-content");

function escapeHTML(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function profileColor(name) {
  let hash = 0;
  for (const char of String(name)) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return PROFILE_COLORS[hash % PROFILE_COLORS.length];
}

function number(value, digits = 0) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(Number(value || 0));
}

function percent(value, digits = 0) {
  return `${number(Number(value || 0) * 100, digits)}%`;
}

function mean(values) {
  const valid = values.filter((value) => Number.isFinite(Number(value)));
  return valid.length
    ? valid.reduce((sum, value) => sum + Number(value), 0) / valid.length
    : 0;
}

async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  return response.json();
}

async function getSeasonData(season) {
  if (!appState.seasonCache.has(season)) {
    appState.seasonCache.set(
      season,
      getJSON(`${DATA_ROOT}/seasons/${season}.json`),
    );
  }
  return appState.seasonCache.get(season);
}

function cleanupCanvases() {
  appState.canvasCleanups.forEach((cleanup) => cleanup());
  appState.canvasCleanups = [];
}

function loading(label = "Loading the model…") {
  main.innerHTML = `
    <div class="loading-state" role="status" aria-live="polite">
      <span class="loading-mark" aria-hidden="true"></span>
      <p>${escapeHTML(label)}</p>
    </div>
  `;
}

function showError(error) {
  console.error(error);
  main.innerHTML = `
    <div class="error-state">
      <p class="eyebrow">Data unavailable</p>
      <h2>The model could not load.</h2>
      <p>Refresh the page to try again. If the issue continues, the generated data bundle may need to be rebuilt.</p>
    </div>
  `;
}

function pageHeader(kicker, title, lede, controls = "") {
  return `
    <header class="page-head">
      <div class="page-head-copy">
        <p class="eyebrow">${escapeHTML(kicker)}</p>
        <h1>${escapeHTML(title)}</h1>
        <p class="lede">${escapeHTML(lede)}</p>
      </div>
      ${controls ? `<div class="controls">${controls}</div>` : ""}
    </header>
  `;
}

function groupControl(group, target = "season-group") {
  return `
    <div class="field">
      <span class="field-label">Position group</span>
      <div class="segmented" data-group-control="${escapeHTML(target)}">
        <button type="button" data-value="forwards" aria-pressed="${group === "forwards"}">Forwards</button>
        <button type="button" data-value="defense" aria-pressed="${group === "defense"}">Defense</button>
      </div>
    </div>
  `;
}

function profileChip(name) {
  return `
    <span class="profile-chip" style="--profile:${profileColor(name)}">
      ${escapeHTML(name)}
    </span>
  `;
}

function metric(label, value, note = "") {
  return `
    <div class="metric">
      <span class="metric-label">${escapeHTML(label)}</span>
      <span class="metric-value">${escapeHTML(value)}</span>
      ${note ? `<span class="metric-note">${escapeHTML(note)}</span>` : ""}
    </div>
  `;
}

function tabs(items, selected, target) {
  return `
    <div class="tab-bar" role="tablist" aria-label="${escapeHTML(target)} views" data-tabs="${escapeHTML(target)}">
      ${items
        .map(
          ([value, label]) => `
            <button
              class="tab-button"
              type="button"
              role="tab"
              data-value="${escapeHTML(value)}"
              aria-selected="${selected === value}"
            >${escapeHTML(label)}</button>
          `,
        )
        .join("")}
    </div>
  `;
}

function bindGroupControl(target, handler) {
  document
    .querySelector(`[data-group-control="${target}"]`)
    ?.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (button) handler(button.dataset.value);
    });
}

function bindTabs(target, handler) {
  document
    .querySelector(`[data-tabs="${target}"]`)
    ?.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (button) handler(button.dataset.value);
    });
}

function setupLineChart(canvas, series, options = {}) {
  if (!canvas) return;
  const context = canvas.getContext("2d");
  const wrapper = canvas.parentElement;
  const colors = getComputedStyle(document.documentElement);
  const grid = colors.getPropertyValue("--line").trim();
  const muted = colors.getPropertyValue("--muted").trim();
  const paper = colors.getPropertyValue("--paper-strong").trim();
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function draw() {
    const width = Math.max(wrapper.clientWidth, 280);
    const height = Number(options.height || 280);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const padding = { top: 16, right: 16, bottom: 38, left: 44 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const allValues = series.flatMap((item) => item.values.map((point) => point.value));
    const min = options.min ?? Math.floor(Math.min(...allValues) / 10) * 10;
    const max = options.max ?? Math.ceil(Math.max(...allValues) / 10) * 10;
    const range = Math.max(max - min, 1);
    const labels = series[0]?.values.map((point) => point.label) || [];

    context.font = "10px Inter, system-ui, sans-serif";
    context.textBaseline = "middle";
    context.strokeStyle = grid;
    context.fillStyle = muted;
    context.lineWidth = 1;

    for (let index = 0; index <= 4; index += 1) {
      const y = padding.top + (plotHeight * index) / 4;
      const value = max - (range * index) / 4;
      context.beginPath();
      context.moveTo(padding.left, y);
      context.lineTo(width - padding.right, y);
      context.stroke();
      context.textAlign = "right";
      context.fillText(`${number(value, 0)}${options.unit || ""}`, padding.left - 9, y);
    }

    const xAt = (index) =>
      padding.left +
      (labels.length <= 1 ? plotWidth / 2 : (plotWidth * index) / (labels.length - 1));
    const yAt = (value) =>
      padding.top + plotHeight - ((value - min) / range) * plotHeight;

    const labelEvery = width < 520 ? 4 : width < 760 ? 3 : 2;
    labels.forEach((label, index) => {
      if (index % labelEvery !== 0 && index !== labels.length - 1) return;
      context.textAlign = "center";
      context.fillStyle = muted;
      context.fillText(label.replace("–", "–"), xAt(index), height - 14);
    });

    series.forEach((item) => {
      context.beginPath();
      context.strokeStyle = item.color;
      context.lineWidth = 2.5;
      context.lineJoin = "round";
      item.values.forEach((point, index) => {
        const x = xAt(index);
        const y = yAt(point.value);
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();

      item.values.forEach((point, index) => {
        if (
          item.highlight &&
          !item.highlight(point, index) &&
          index !== item.values.length - 1
        ) {
          return;
        }
        context.beginPath();
        context.fillStyle = paper;
        context.strokeStyle = item.color;
        context.lineWidth = 2;
        context.arc(xAt(index), yAt(point.value), 3.7, 0, Math.PI * 2);
        context.fill();
        context.stroke();
      });
    });

    if (!reducedMotion) {
      canvas.animate(
        [{ opacity: 0.86 }, { opacity: 1 }],
        { duration: 180, easing: "ease-out" },
      );
    }
  }

  draw();
  const observer = new ResizeObserver(draw);
  observer.observe(wrapper);
  appState.canvasCleanups.push(() => observer.disconnect());
}

function renderOverview() {
  const { meta } = appState.core;
  const oldest = meta.seasons.at(-1).label;
  const latest = meta.seasons[0].label;
  const forwardSwitch = Math.round(meta.switchRates.forwards * 100);
  const defenseSwitch = Math.round(meta.switchRates.defense * 100);

  main.innerHTML = `
    <article class="page">
      <section class="hero">
        <div class="hero-copy">
          <p class="eyebrow">NHL PLAYER STYLE MODEL · ${escapeHTML(oldest)}–${escapeHTML(latest)}</p>
          <h1>See how players play, <span class="hero-accent">not just what they score.</span></h1>
          <p class="lede">Explore role, identity, and change across seasons, teams, careers, and playoff runs.</p>
          <div class="hero-actions">
            <a class="button button-primary" href="#season">Explore a season</a>
            <a class="button" href="#glossary">Browse styles</a>
          </div>
        </div>
        <div class="hero-board" aria-label="${number(meta.playerCount)} players across ${meta.seasonCount} seasons">
          <span class="board-center" aria-hidden="true"></span>
          <span class="board-puck" aria-hidden="true"></span>
          <span class="board-stat board-stat-top">
            <strong>${number(meta.playerCount)}</strong>
            <span>players</span>
          </span>
          <span class="board-stat board-stat-bottom">
            <strong>${meta.seasonCount}</strong>
            <span>seasons</span>
          </span>
        </div>
      </section>

      <section class="metric-grid" aria-label="Dataset summary">
        ${metric("Season coverage", `${oldest}–${latest}`)}
        ${metric("Player-seasons", number(meta.playerSeasonCount), `${number(Object.values(meta.profileDefinitions).reduce((a, b) => a + b, 0))} profiles learned`)}
        ${metric("Forward switch rate", `${forwardSwitch}%`, "median year-to-year change")}
        ${metric("Defense switch rate", `${defenseSwitch}%`, "median year-to-year change")}
      </section>

      <section class="analysis-grid">
        <div class="chart-panel">
          <div class="chart-title-row">
            <div>
              <h3>Model confidence by season</h3>
              <p>Average probability assigned to each player’s top style.</p>
            </div>
            <div class="legend" aria-label="Chart legend">
              <span><i style="--legend-color:var(--aqua)"></i>Forwards</span>
              <span><i style="--legend-color:var(--coral)"></i>Defense</span>
            </div>
          </div>
          <div class="canvas-wrap">
            <canvas id="confidence-chart" role="img" aria-label="Average model confidence for forwards and defense by season"></canvas>
          </div>
        </div>
        <div class="info-panel">
          <p class="eyebrow">What changes</p>
          <div class="insight-stack">
            <div class="insight">
              <strong>${forwardSwitch}%</strong>
              <span>Median forward style changes between adjacent seasons.</span>
            </div>
            <div class="insight">
              <strong>${defenseSwitch}%</strong>
              <span>Median defense style changes between adjacent seasons.</span>
            </div>
            <div class="insight">
              <strong>Soft fit</strong>
              <span>Every player keeps a probability mix, not a forced label.</span>
            </div>
          </div>
        </div>
      </section>

      <section>
        <div class="section-heading">
          <div>
            <p class="eyebrow">The model</p>
            <h2>From stat line to style mix</h2>
          </div>
          <p>Public NHL and MoneyPuck data becomes a season-specific probability profile.</p>
        </div>
        <div class="method-grid">
          <article class="method-step" style="--step-color:var(--aqua)">
            <span>01 · RATE</span>
            <h3>Normalize usage</h3>
            <p>Convert counting stats to per-60 rates and usage shares.</p>
          </article>
          <article class="method-step" style="--step-color:var(--coral)">
            <span>02 · SCALE</span>
            <h3>Balance signals</h3>
            <p>Robust scaling keeps extreme values from owning the result.</p>
          </article>
          <article class="method-step" style="--step-color:var(--gold)">
            <span>03 · COMPRESS</span>
            <h3>Build fingerprints</h3>
            <p>NMF reduces correlated stats into compact style components.</p>
          </article>
          <article class="method-step" style="--step-color:#5f8dd3">
            <span>04 · CLUSTER</span>
            <h3>Assign probabilities</h3>
            <p>A mixture model returns a ranked set of style fits per player.</p>
          </article>
        </div>
      </section>

      <footer class="page-footer">
        <span>Sources: NHL Gamecenter + MoneyPuck</span>
        <span>Advanced-data era begins in 2008–09</span>
      </footer>
    </article>
  `;

  const trend = meta.confidenceTrend;
  setupLineChart(
    document.querySelector("#confidence-chart"),
    [
      {
        color: "#21b6a8",
        values: trend.map((row) => ({ label: row.label, value: row.forwards })),
      },
      {
        color: "#f06445",
        values: trend.map((row) => ({ label: row.label, value: row.defense })),
      },
    ],
    { min: 60, max: 100, unit: "%" },
  );
}

function glossaryRows() {
  const query = appState.glossaryQuery.trim().toLowerCase();
  return appState.core.glossary[appState.glossaryGroup].filter((row) => {
    if (!query) return true;
    const haystack = [
      row.name,
      row.description,
      ...row.examples,
      ...row.high.map((trait) => trait.label),
      ...row.low.map((trait) => trait.label),
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });
}

function updateGlossaryList() {
  const rows = glossaryRows();
  const count = document.querySelector("#glossary-count");
  const list = document.querySelector("#glossary-list");
  if (!count || !list) return;
  count.textContent = `${rows.length} style${rows.length === 1 ? "" : "s"}`;
  list.innerHTML = rows.length
    ? rows
        .map(
          (row) => `
            <article class="profile-row" style="--profile:${profileColor(row.name)}">
              <div>
                <h3 class="profile-name">${escapeHTML(row.name)}</h3>
                <p>${escapeHTML(row.description)}</p>
              </div>
              <div class="trait-columns">
                <div class="trait-column">
                  <strong>Strong signals</strong>
                  <ul>
                    ${row.high
                      .map(
                        (trait) =>
                          `<li class="trait-up">↑ ${escapeHTML(trait.label)} · ${number(Math.abs(trait.z), 1)}σ</li>`,
                      )
                      .join("")}
                  </ul>
                </div>
                <div class="trait-column">
                  <strong>Lower signals</strong>
                  <ul>
                    ${row.low
                      .map(
                        (trait) =>
                          `<li class="trait-down">↓ ${escapeHTML(trait.label)} · ${number(Math.abs(trait.z), 1)}σ</li>`,
                      )
                      .join("")}
                  </ul>
                </div>
              </div>
              <div class="examples">
                <strong>Recent examples</strong>
                ${row.examples.map(escapeHTML).join(" · ") || "No recent examples"}
              </div>
            </article>
          `,
        )
        .join("")
    : `<div class="empty-state">No styles match that search.</div>`;
}

function renderGlossary() {
  main.innerHTML = `
    <article class="page">
      ${pageHeader(
        "Style glossary",
        "Every player style, in plain language.",
        "See the model’s defining signals and recent examples without decoding raw feature names.",
        groupControl(appState.glossaryGroup, "glossary-group"),
      )}
      <div class="glossary-toolbar">
        <div class="field" style="width:min(100%, 380px)">
          <label for="glossary-search">Find a style or player</label>
          <input
            class="search-input"
            id="glossary-search"
            type="search"
            value="${escapeHTML(appState.glossaryQuery)}"
            placeholder="Try playmaker, shutdown, McDavid…"
          />
        </div>
        <span class="result-count" id="glossary-count"></span>
      </div>
      <section class="glossary-list" id="glossary-list" aria-live="polite"></section>
      <footer class="page-footer">
        <span>Profiles are learned independently each season.</span>
        <span>Examples show strong recent matches.</span>
      </footer>
    </article>
  `;

  bindGroupControl("glossary-group", (group) => {
    appState.glossaryGroup = group;
    renderGlossary();
  });
  document.querySelector("#glossary-search")?.addEventListener("input", (event) => {
    appState.glossaryQuery = event.target.value;
    updateGlossaryList();
  });
  updateGlossaryList();
}

function seasonControls() {
  const seasonOptions = appState.core.meta.seasons
    .map(
      (season) =>
        `<option value="${season.key}" ${season.key === appState.season ? "selected" : ""}>${escapeHTML(season.label)}</option>`,
    )
    .join("");
  return `
    <div class="field">
      <label for="season-select">Season</label>
      <select id="season-select">${seasonOptions}</select>
    </div>
    ${groupControl(appState.group, "season-group")}
  `;
}

function profileBars(profiles, limit = 10) {
  const maxShare = Math.max(...profiles.map((profile) => profile.share), 1);
  return `
    <div class="bar-list">
      ${profiles
        .slice(0, limit)
        .map(
          (profile) => `
            <div class="bar-row">
              <span class="bar-label">${escapeHTML(profile.name)}</span>
              <span class="bar-track" aria-hidden="true">
                <span
                  class="bar-fill"
                  style="width:${(profile.share / maxShare) * 100}%;--profile:${profileColor(profile.name)}"
                ></span>
              </span>
              <span class="bar-value">${number(profile.share, 1)}%</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderSeasonSnapshot(groupData) {
  const dominant = groupData.profiles[0];
  const topThree = groupData.profiles
    .slice(0, 3)
    .reduce((sum, profile) => sum + profile.share, 0);
  return `
    <section class="metric-grid">
      ${metric("Players", number(groupData.players.length))}
      ${metric("Dominant style", `${number(dominant.share, 1)}%`, dominant.name)}
      ${metric("Top-three share", `${number(topThree, 1)}%`)}
      ${metric("Mixed profiles", number(groupData.mixedCount), "below 80% top-style confidence")}
    </section>
    <section class="analysis-grid">
      <div class="chart-panel">
        <div class="chart-title-row">
          <div>
            <h3>Roster-wide style mix</h3>
            <p>Share of player-seasons in each top profile.</p>
          </div>
        </div>
        ${profileBars(groupData.profiles)}
      </div>
      <div class="info-panel">
        <p class="eyebrow">Season read</p>
        <div class="insight-stack">
          <div class="insight">
            <strong>${percent(groupData.averageConfidence, 0)}</strong>
            <span>Average top-style confidence.</span>
          </div>
          <div class="insight">
            <strong>${groupData.profiles.length}</strong>
            <span>Distinct profiles present.</span>
          </div>
          <div class="insight">
            <strong>${escapeHTML(dominant.name)}</strong>
            <span>Most common top assignment.</span>
          </div>
        </div>
      </div>
    </section>
  `;
}

function playerRows(players, query) {
  const normalized = query.trim().toLowerCase();
  return players
    .filter((player) => !normalized || player.name.toLowerCase().includes(normalized))
    .slice(0, 80);
}

function playerDetail(player) {
  if (!player) {
    return `<div class="detail-panel player-detail empty-state">Select a player to inspect their style mix.</div>`;
  }
  return `
    <aside class="detail-panel player-detail" aria-label="${escapeHTML(player.name)} detail">
      <p class="detail-kicker">Player profile</p>
      <div class="detail-name">${escapeHTML(player.name)}</div>
      <p class="detail-meta">${escapeHTML(player.position)} · ${escapeHTML(player.team)} · ${number(player.games)} games</p>
      ${profileChip(player.profile)}
      <div class="detail-stats">
        <div class="detail-stat"><span>Points</span><strong>${number(player.points)}</strong></div>
        <div class="detail-stat"><span>TOI</span><strong>${number(player.toi, 1)}</strong></div>
        <div class="detail-stat"><span>Fit</span><strong>${percent(player.confidence)}</strong></div>
      </div>
      <div class="probability-list">
        ${player.probabilities
          .map(
            (item) => `
              <div class="probability-row">
                <div>
                  <div class="probability-label">${escapeHTML(item.profile)}</div>
                  <div class="bar-track">
                    <div class="bar-fill" style="width:${item.value * 100}%;--profile:${profileColor(item.profile)}"></div>
                  </div>
                </div>
                <span class="probability-value">${percent(item.value)}</span>
              </div>
            `,
          )
          .join("")}
      </div>
    </aside>
  `;
}

function renderPlayerTable(players, selectedId, interactive = true) {
  return `
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Player</th>
            <th>Style</th>
            <th class="numeric">GP</th>
            <th class="numeric">PTS</th>
            <th class="numeric">Fit</th>
          </tr>
        </thead>
        <tbody>
          ${players
            .map(
              (player) => `
                <tr ${player.id === selectedId ? 'aria-current="true"' : ""}>
                  <td>
                    ${
                      interactive
                        ? `<button class="text-button" type="button" data-player-id="${player.id}">${escapeHTML(player.name)}</button>`
                        : `<strong>${escapeHTML(player.name)}</strong>`
                    }
                    <br><span class="result-count">${escapeHTML(player.team)} · ${escapeHTML(player.position)}</span>
                  </td>
                  <td>${profileChip(player.profile)}</td>
                  <td class="numeric">${number(player.games)}</td>
                  <td class="numeric">${number(player.points)}</td>
                  <td class="numeric">${percent(player.confidence)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderSeasonPlayers(groupData) {
  const initialPlayers = playerRows(groupData.players, "");
  if (!appState.selectedPlayerId && initialPlayers[0]) {
    appState.selectedPlayerId = initialPlayers[0].id;
  }
  const selected = groupData.players.find(
    (player) => player.id === appState.selectedPlayerId,
  );
  return `
    <div class="glossary-toolbar">
      <div class="field" style="width:min(100%, 360px)">
        <label for="player-search">Search players</label>
        <input class="search-input" id="player-search" type="search" placeholder="Player name" />
      </div>
      <span class="result-count">Select a name for the full probability mix.</span>
    </div>
    <div class="two-column">
      <div id="player-table">${renderPlayerTable(initialPlayers, appState.selectedPlayerId)}</div>
      <div id="player-detail">${playerDetail(selected)}</div>
    </div>
  `;
}

function renderSeasonTeams(groupData) {
  const teams = [...new Set(
    groupData.players.flatMap((player) => player.team.split("/").filter(Boolean)),
  )].sort();
  const team = teams[0] || "";
  return `
    <div class="glossary-toolbar">
      <div class="field">
        <label for="team-select">Team</label>
        <select id="team-select">
          ${teams.map((value) => `<option value="${escapeHTML(value)}">${escapeHTML(value)}</option>`).join("")}
        </select>
      </div>
      <span class="result-count">Style coverage across the selected roster.</span>
    </div>
    <div id="team-view">${teamView(groupData.players, team)}</div>
  `;
}

function teamView(players, team) {
  const roster = players
    .filter((player) => player.team.split("/").includes(team))
    .sort((a, b) => b.points - a.points);
  const counts = Counter(roster.map((player) => player.profile));
  const profiles = [...counts.entries()]
    .map(([name, count]) => ({
      name,
      count,
      share: roster.length ? (count / roster.length) * 100 : 0,
    }))
    .sort((a, b) => b.count - a.count);
  return `
    <div class="two-column">
      <div>
        ${renderPlayerTable(roster, null, false)}
      </div>
      <div class="chart-panel">
        <div class="chart-title-row"><div><h3>${escapeHTML(team)} style mix</h3><p>${roster.length} players in this group.</p></div></div>
        ${profileBars(profiles, 8)}
      </div>
    </div>
  `;
}

function Counter(values) {
  const counts = new Map();
  values.forEach((value) => counts.set(value, (counts.get(value) || 0) + 1));
  return counts;
}

function renderNeedFinder(groupData) {
  const profiles = groupData.profiles.map((profile) => profile.name);
  const teams = [...new Set(
    groupData.players.flatMap((player) => player.team.split("/").filter(Boolean)),
  )].sort();
  const target = profiles[0] || "";
  return `
    <div class="controls" style="margin-bottom:24px">
      <div class="field">
        <label for="need-profile">Target style</label>
        <select id="need-profile">${profiles.map((name) => `<option value="${escapeHTML(name)}">${escapeHTML(name)}</option>`).join("")}</select>
      </div>
      <div class="field">
        <label for="need-team">Exclude team</label>
        <select id="need-team"><option value="">None</option>${teams.map((team) => `<option value="${escapeHTML(team)}">${escapeHTML(team)}</option>`).join("")}</select>
      </div>
      <div class="field">
        <label for="need-games">Minimum games · <span id="need-games-value">20</span></label>
        <input id="need-games" type="range" min="0" max="82" value="20" step="5" />
      </div>
    </div>
    <div id="need-results">${needResults(groupData.players, target, "", 20)}</div>
  `;
}

function needResults(players, target, excludedTeam, minGames) {
  const matches = players
    .map((player) => ({
      ...player,
      targetFit:
        player.probabilities.find((item) => item.profile === target)?.value || 0,
    }))
    .filter(
      (player) =>
        player.games >= minGames &&
        !player.team.split("/").includes(excludedTeam) &&
        player.targetFit > 0,
    )
    .sort((a, b) => b.targetFit - a.targetFit)
    .slice(0, 30);
  if (!matches.length) {
    return `<div class="empty-state">No players match those filters.</div>`;
  }
  return `
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>Player</th><th>Team</th><th>Top style</th><th class="numeric">Target fit</th><th class="numeric">GP</th><th class="numeric">PTS</th></tr></thead>
        <tbody>
          ${matches
            .map(
              (player) => `
                <tr>
                  <td><strong>${escapeHTML(player.name)}</strong><br><span class="result-count">${escapeHTML(player.position)}</span></td>
                  <td>${escapeHTML(player.team)}</td>
                  <td>${profileChip(player.profile)}</td>
                  <td class="numeric">${percent(player.targetFit)}</td>
                  <td class="numeric">${number(player.games)}</td>
                  <td class="numeric">${number(player.points)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

async function renderSeason() {
  const seasonLabel =
    appState.core.meta.seasons.find((item) => item.key === appState.season)?.label ||
    appState.season;
  main.innerHTML = `
    <article class="page">
      ${pageHeader(
        "Season lab",
        "How was the league built?",
        "Move from league-wide style mix to a player, roster, or acquisition need.",
        seasonControls(),
      )}
      <div id="season-content">
        <div class="loading-state" style="min-height:45vh"><span class="loading-mark"></span><p>Loading ${escapeHTML(seasonLabel)}…</p></div>
      </div>
    </article>
  `;

  bindGroupControl("season-group", (group) => {
    appState.group = group;
    appState.selectedPlayerId = null;
    renderSeason();
  });
  document.querySelector("#season-select")?.addEventListener("change", (event) => {
    appState.season = event.target.value;
    appState.selectedPlayerId = null;
    renderSeason();
  });

  const data = await getSeasonData(appState.season);
  if (appState.route !== "season") return;
  const groupData = data[appState.group];
  const content = document.querySelector("#season-content");
  content.innerHTML = `
    ${tabs(
      [
        ["snapshot", "Snapshot"],
        ["players", "Players"],
        ["teams", "Teams"],
        ["needs", "Need finder"],
      ],
      appState.seasonTab,
      "season",
    )}
    <section id="season-tab-panel"></section>
    <footer class="page-footer">
      <span>${escapeHTML(seasonLabel)} · ${appState.group === "forwards" ? "Forwards" : "Defense"}</span>
      <span>Style fit uses regular-season data.</span>
    </footer>
  `;

  const panel = document.querySelector("#season-tab-panel");
  const drawTab = () => {
    if (appState.seasonTab === "snapshot") {
      panel.innerHTML = renderSeasonSnapshot(groupData);
    } else if (appState.seasonTab === "players") {
      panel.innerHTML = renderSeasonPlayers(groupData);
      bindSeasonPlayerExplorer(groupData);
    } else if (appState.seasonTab === "teams") {
      panel.innerHTML = renderSeasonTeams(groupData);
      document.querySelector("#team-select")?.addEventListener("change", (event) => {
        document.querySelector("#team-view").innerHTML = teamView(
          groupData.players,
          event.target.value,
        );
      });
    } else {
      panel.innerHTML = renderNeedFinder(groupData);
      bindNeedFinder(groupData);
    }
  };
  bindTabs("season", (tab) => {
    appState.seasonTab = tab;
    renderSeason();
  });
  drawTab();
}

function bindSeasonPlayerExplorer(groupData) {
  const search = document.querySelector("#player-search");
  const table = document.querySelector("#player-table");
  const detail = document.querySelector("#player-detail");
  const drawRows = () => {
    const players = playerRows(groupData.players, search.value);
    table.innerHTML = renderPlayerTable(players, appState.selectedPlayerId);
  };
  search?.addEventListener("input", drawRows);
  table?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-player-id]");
    if (!button) return;
    appState.selectedPlayerId = Number(button.dataset.playerId);
    const player = groupData.players.find(
      (item) => item.id === appState.selectedPlayerId,
    );
    detail.innerHTML = playerDetail(player);
    drawRows();
  });
}

function bindNeedFinder(groupData) {
  const profile = document.querySelector("#need-profile");
  const team = document.querySelector("#need-team");
  const games = document.querySelector("#need-games");
  const results = document.querySelector("#need-results");
  const draw = () => {
    document.querySelector("#need-games-value").textContent = games.value;
    results.innerHTML = needResults(
      groupData.players,
      profile.value,
      team.value,
      Number(games.value),
    );
  };
  [profile, team, games].forEach((control) => control?.addEventListener("input", draw));
}

function careerPlayers(records, group) {
  const grouped = new Map();
  records
    .filter((record) => record.group === group)
    .forEach((record) => {
      if (!grouped.has(record.id)) {
        grouped.set(record.id, {
          id: record.id,
          name: record.name,
          position: record.position,
          seasons: 0,
        });
      }
      grouped.get(record.id).seasons += 1;
    });
  return [...grouped.values()].sort(
    (a, b) => b.seasons - a.seasons || a.name.localeCompare(b.name),
  );
}

function careerSearchResults(players, query) {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return players.slice(0, 7);
  return players
    .filter((player) => player.name.toLowerCase().includes(normalized))
    .slice(0, 8);
}

function careerView(history) {
  const sorted = [...history].sort((a, b) => a.season.localeCompare(b.season));
  const switches = sorted.filter(
    (row, index) => index > 0 && row.profile !== sorted[index - 1].profile,
  ).length;
  const common = [...Counter(sorted.map((row) => row.profile)).entries()].sort(
    (a, b) => b[1] - a[1],
  )[0]?.[0];
  return `
    <section class="metric-grid">
      ${metric("Seasons", number(sorted.length))}
      ${metric("Average fit", percent(mean(sorted.map((row) => row.confidence))))}
      ${metric("Style changes", number(switches))}
      ${metric("Most common", common || "—")}
    </section>
    <section class="chart-panel" style="margin-bottom:24px">
      <div class="chart-title-row">
        <div><h3>Top-style confidence</h3><p>Changes are marked where the leading profile switches.</p></div>
      </div>
      <div class="canvas-wrap">
        <canvas id="career-chart" role="img" aria-label="Top style confidence by season"></canvas>
      </div>
    </section>
    <div class="table-wrap timeline-table">
      <table class="data-table">
        <thead><tr><th>Season</th><th>Style</th><th class="numeric">Fit</th><th>Team</th><th class="numeric">GP</th><th class="numeric">PTS</th><th class="numeric">TOI</th></tr></thead>
        <tbody>
          ${sorted
            .map(
              (row) => `
                <tr>
                  <td>${escapeHTML(appState.core.meta.seasons.find((item) => item.key === row.season)?.label || row.season)}</td>
                  <td class="profile-cell">${profileChip(row.profile)}</td>
                  <td class="numeric">${percent(row.confidence)}</td>
                  <td>${escapeHTML(row.team)}</td>
                  <td class="numeric">${number(row.games)}</td>
                  <td class="numeric">${number(row.points)}</td>
                  <td class="numeric">${number(row.toi, 1)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

async function renderCareer() {
  if (!appState.careers) {
    loading("Loading career histories…");
    appState.careers = await getJSON(`${DATA_ROOT}/careers.json`);
  }
  if (appState.route !== "career") return;
  const players = careerPlayers(appState.careers, appState.careerGroup);
  if (!appState.careerPlayerId || !players.some((player) => player.id === appState.careerPlayerId)) {
    appState.careerPlayerId = players[0]?.id;
  }
  const selected = players.find((player) => player.id === appState.careerPlayerId);
  const history = appState.careers.filter(
    (record) =>
      record.group === appState.careerGroup &&
      record.id === appState.careerPlayerId,
  );

  main.innerHTML = `
    <article class="page">
      ${pageHeader(
        "Career paths",
        "Style is a timeline, not a label.",
        "Track how a player’s role, confidence, and production move from season to season.",
        groupControl(appState.careerGroup, "career-group"),
      )}
      <div class="two-column" style="margin-bottom:28px">
        <div>
          <div class="field">
            <label for="career-search">Find a player</label>
            <input
              class="search-input"
              id="career-search"
              type="search"
              value="${escapeHTML(appState.careerQuery)}"
              placeholder="Search by name"
              autocomplete="off"
            />
          </div>
          <div class="search-results" id="career-search-results"></div>
        </div>
        <div class="detail-panel">
          <p class="detail-kicker">Selected player</p>
          <div class="detail-name">${escapeHTML(selected?.name || "No player")}</div>
          <p class="detail-meta">${escapeHTML(selected?.position || "")} · ${number(selected?.seasons || 0)} seasons</p>
        </div>
      </div>
      <section id="career-view">${careerView(history)}</section>
      <footer class="page-footer">
        <span>Styles are re-learned each season.</span>
        <span>A switch means the top probability changed.</span>
      </footer>
    </article>
  `;

  bindGroupControl("career-group", (group) => {
    appState.careerGroup = group;
    appState.careerPlayerId = null;
    appState.careerQuery = "";
    renderCareer();
  });
  const search = document.querySelector("#career-search");
  const results = document.querySelector("#career-search-results");
  const drawSearch = () => {
    appState.careerQuery = search.value;
    results.innerHTML = careerSearchResults(players, search.value)
      .map(
        (player) => `
          <button class="search-result" type="button" data-career-id="${player.id}">
            <span>${escapeHTML(player.name)}</span>
            <span>${player.seasons} seasons</span>
          </button>
        `,
      )
      .join("");
  };
  search.addEventListener("input", drawSearch);
  results.addEventListener("click", (event) => {
    const button = event.target.closest("[data-career-id]");
    if (!button) return;
    appState.careerPlayerId = Number(button.dataset.careerId);
    appState.careerQuery = "";
    renderCareer();
  });
  drawSearch();

  const sortedHistory = [...history].sort((a, b) => a.season.localeCompare(b.season));
  setupLineChart(
    document.querySelector("#career-chart"),
    [
      {
        color: "#21b6a8",
        values: sortedHistory.map((row) => ({
          label:
            appState.core.meta.seasons.find((item) => item.key === row.season)
              ?.label || row.season,
          value: row.confidence * 100,
          profile: row.profile,
        })),
        highlight: (point, index) =>
          index > 0 && point.profile !== sortedHistory[index - 1].profile,
      },
    ],
    { min: 0, max: 100, unit: "%" },
  );
}

function playoffFiltered() {
  return appState.playoffs.filter(
    (row) =>
      row.season === appState.playoffSeason &&
      row.group === appState.playoffGroup &&
      row.regGames >= appState.playoffMinReg &&
      row.playoffGames >= appState.playoffMinPo,
  );
}

function playoffControls() {
  const available = [...new Set(appState.playoffs.map((row) => row.season))]
    .sort()
    .reverse();
  return `
    <div class="field">
      <label for="playoff-season">Season</label>
      <select id="playoff-season">
        ${available
          .map((season) => {
            const label =
              appState.core.meta.seasons.find((item) => item.key === season)?.label ||
              season;
            return `<option value="${season}" ${season === appState.playoffSeason ? "selected" : ""}>${escapeHTML(label)}</option>`;
          })
          .join("")}
      </select>
    </div>
    ${groupControl(appState.playoffGroup, "playoff-group")}
  `;
}

function playoffShiftTable(rows) {
  const sorted = [...rows].sort((a, b) => b.distance - a.distance).slice(0, 35);
  return `
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>Player</th><th>Regular season</th><th>Playoffs</th><th class="numeric">Shift</th><th class="numeric">PO GP</th></tr></thead>
        <tbody>
          ${sorted
            .map(
              (row) => `
                <tr>
                  <td><strong>${escapeHTML(row.name)}</strong><br><span class="result-count">${escapeHTML(row.team)} · ${escapeHTML(row.position)}</span></td>
                  <td>${profileChip(row.regProfile)}</td>
                  <td>${profileChip(row.playoffProfile)}</td>
                  <td class="numeric">${number(row.distance, 2)}</td>
                  <td class="numeric">${number(row.playoffGames)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function playoffShiftView(rows) {
  const changed = rows.filter((row) => row.changed).length;
  const changeRate = rows.length ? changed / rows.length : 0;
  return `
    <section class="metric-grid">
      ${metric("Qualified players", number(rows.length))}
      ${metric("Top-style changes", percent(changeRate))}
      ${metric("Average shift", number(mean(rows.map((row) => row.distance)), 2))}
      ${metric("Playoff scoring", number(mean(rows.map((row) => row.playoffPpg)), 2), "points per game")}
    </section>
    <div class="section-heading">
      <div><p class="eyebrow">Largest movement</p><h2>Who changed most?</h2></div>
      <p>Shift compares the full regular-season and playoff probability mixes.</p>
    </div>
    ${rows.length ? playoffShiftTable(rows) : '<div class="empty-state">No players match these filters.</div>'}
  `;
}

function playoffTransitions(rows) {
  const transitions = Counter(
    rows
      .filter((row) => row.changed)
      .map((row) => `${row.regProfile}|||${row.playoffProfile}`),
  );
  const sorted = [...transitions.entries()].sort((a, b) => b[1] - a[1]).slice(0, 18);
  return `
    <div class="section-heading">
      <div><p class="eyebrow">Style movement</p><h2>Common playoff transitions</h2></div>
      <p>Only players whose leading profile changed are shown.</p>
    </div>
    <div class="transition-list">
      ${sorted.length
        ? sorted
            .map(([key, count]) => {
              const [from, to] = key.split("|||");
              return `
                <div class="transition-row">
                  ${profileChip(from)}
                  <span class="transition-arrow">→</span>
                  ${profileChip(to)}
                  <span class="transition-count">${count} player${count === 1 ? "" : "s"}</span>
                </div>
              `;
            })
            .join("")
        : '<div class="empty-state">No style transitions match these filters.</div>'}
    </div>
  `;
}

function playoffPlayerView(rows) {
  const players = [...new Map(
    appState.playoffs
      .filter((row) => row.group === appState.playoffGroup)
      .map((row) => [row.id, { id: row.id, name: row.name }]),
  ).values()].sort((a, b) => a.name.localeCompare(b.name));
  if (!appState.playoffPlayerId || !players.some((player) => player.id === appState.playoffPlayerId)) {
    appState.playoffPlayerId = rows[0]?.id || players[0]?.id;
  }
  const history = appState.playoffs
    .filter(
      (row) =>
        row.group === appState.playoffGroup &&
        row.id === appState.playoffPlayerId &&
        row.playoffGames >= 1,
    )
    .sort((a, b) => a.season.localeCompare(b.season));
  const selected = players.find((player) => player.id === appState.playoffPlayerId);
  return `
    <div class="two-column" style="margin-bottom:28px">
      <div>
        <div class="field">
          <label for="playoff-player-search">Find a player</label>
          <input class="search-input" id="playoff-player-search" type="search" value="${escapeHTML(appState.playoffQuery)}" placeholder="Search playoff players" />
        </div>
        <div class="search-results" id="playoff-player-results"></div>
      </div>
      <div class="detail-panel">
        <p class="detail-kicker">Selected player</p>
        <div class="detail-name">${escapeHTML(selected?.name || "No player")}</div>
        <p class="detail-meta">${history.length} playoff season${history.length === 1 ? "" : "s"} in the dataset</p>
      </div>
    </div>
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>Season</th><th>Regular season</th><th>Playoffs</th><th class="numeric">REG P/GP</th><th class="numeric">PO P/GP</th><th class="numeric">Shift</th></tr></thead>
        <tbody>
          ${history
            .map(
              (row) => `
                <tr>
                  <td>${escapeHTML(appState.core.meta.seasons.find((item) => item.key === row.season)?.label || row.season)}</td>
                  <td>${profileChip(row.regProfile)}</td>
                  <td>${profileChip(row.playoffProfile)}</td>
                  <td class="numeric">${number(row.regPpg, 2)}</td>
                  <td class="numeric">${number(row.playoffPpg, 2)}</td>
                  <td class="numeric">${number(row.distance, 2)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

async function renderPlayoffs() {
  if (!appState.playoffs) {
    loading("Loading playoff histories…");
    appState.playoffs = await getJSON(`${DATA_ROOT}/playoffs.json`);
    const viable = appState.playoffs
      .filter((row) => row.playoffGames >= 1)
      .map((row) => row.season)
      .sort()
      .reverse();
    appState.playoffSeason = viable[0] || appState.playoffs[0]?.season;
  }
  if (appState.route !== "playoffs") return;
  const rows = playoffFiltered();
  main.innerHTML = `
    <article class="page">
      ${pageHeader(
        "Playoff pressure",
        "What survives the postseason?",
        "Compare regular-season identity with the role a player actually carried in the playoffs.",
        playoffControls(),
      )}
      <div class="controls" style="margin-bottom:22px">
        <div class="field">
          <label for="playoff-reg-games">Minimum regular games · <span id="playoff-reg-value">${appState.playoffMinReg}</span></label>
          <input id="playoff-reg-games" type="range" min="0" max="82" step="5" value="${appState.playoffMinReg}" />
        </div>
        <div class="field">
          <label for="playoff-po-games">Minimum playoff games · <span id="playoff-po-value">${appState.playoffMinPo}</span></label>
          <input id="playoff-po-games" type="range" min="1" max="28" step="1" value="${appState.playoffMinPo}" />
        </div>
      </div>
      ${tabs(
        [
          ["shifts", "Player shifts"],
          ["transitions", "Style transitions"],
          ["player", "Player career"],
        ],
        appState.playoffTab,
        "playoffs",
      )}
      <section id="playoff-panel"></section>
      <footer class="page-footer">
        <span>Playoff profiles use the regular-season model.</span>
        <span>Shift score compares full probability mixes.</span>
      </footer>
    </article>
  `;

  const panel = document.querySelector("#playoff-panel");
  if (appState.playoffTab === "shifts") panel.innerHTML = playoffShiftView(rows);
  else if (appState.playoffTab === "transitions") panel.innerHTML = playoffTransitions(rows);
  else {
    panel.innerHTML = playoffPlayerView(rows);
    bindPlayoffPlayerSearch();
  }

  bindGroupControl("playoff-group", (group) => {
    appState.playoffGroup = group;
    appState.playoffPlayerId = null;
    renderPlayoffs();
  });
  document.querySelector("#playoff-season")?.addEventListener("change", (event) => {
    appState.playoffSeason = event.target.value;
    renderPlayoffs();
  });
  document.querySelector("#playoff-reg-games")?.addEventListener("input", (event) => {
    document.querySelector("#playoff-reg-value").textContent = event.target.value;
  });
  document.querySelector("#playoff-reg-games")?.addEventListener("change", (event) => {
    appState.playoffMinReg = Number(event.target.value);
    renderPlayoffs();
  });
  document.querySelector("#playoff-po-games")?.addEventListener("input", (event) => {
    document.querySelector("#playoff-po-value").textContent = event.target.value;
  });
  document.querySelector("#playoff-po-games")?.addEventListener("change", (event) => {
    appState.playoffMinPo = Number(event.target.value);
    renderPlayoffs();
  });
  bindTabs("playoffs", (tab) => {
    appState.playoffTab = tab;
    renderPlayoffs();
  });
}

function bindPlayoffPlayerSearch() {
  const players = [...new Map(
    appState.playoffs
      .filter((row) => row.group === appState.playoffGroup && row.playoffGames >= 1)
      .map((row) => [row.id, { id: row.id, name: row.name }]),
  ).values()].sort((a, b) => a.name.localeCompare(b.name));
  const search = document.querySelector("#playoff-player-search");
  const results = document.querySelector("#playoff-player-results");
  const draw = () => {
    appState.playoffQuery = search.value;
    const query = search.value.trim().toLowerCase();
    results.innerHTML = players
      .filter((player) => !query || player.name.toLowerCase().includes(query))
      .slice(0, 8)
      .map(
        (player) => `
          <button class="search-result" type="button" data-playoff-player="${player.id}">
            <span>${escapeHTML(player.name)}</span><span>View history</span>
          </button>
        `,
      )
      .join("");
  };
  search?.addEventListener("input", draw);
  results?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-playoff-player]");
    if (!button) return;
    appState.playoffPlayerId = Number(button.dataset.playoffPlayer);
    appState.playoffQuery = "";
    renderPlayoffs();
  });
  draw();
}

async function renderRoute() {
  cleanupCanvases();
  const route = location.hash.replace(/^#\/?/, "") || "overview";
  appState.route = ROUTE_LABELS[route] ? route : "overview";
  document.querySelectorAll("[data-route]").forEach((link) => {
    const active = link.dataset.route === appState.route;
    link.classList.toggle("is-active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.querySelector("#mobile-section").textContent =
    ROUTE_LABELS[appState.route];

  try {
    if (appState.route === "overview") renderOverview();
    else if (appState.route === "glossary") renderGlossary();
    else if (appState.route === "season") await renderSeason();
    else if (appState.route === "career") await renderCareer();
    else if (appState.route === "playoffs") await renderPlayoffs();
  } catch (error) {
    showError(error);
  }
  document.title = `${ROUTE_LABELS[appState.route]} · NHL Player Style Lab`;
  window.scrollTo({ top: 0, behavior: "auto" });
}

async function init() {
  try {
    appState.core = await getJSON(`${DATA_ROOT}/core.json`);
    appState.season = appState.core.meta.seasons[0].key;
    const oldest = appState.core.meta.seasons.at(-1).label;
    const latest = appState.core.meta.seasons[0].label;
    document.querySelector("#coverage-label").textContent =
      `${oldest}–${latest} · public data`;
    await renderRoute();
  } catch (error) {
    showError(error);
  }
}

window.addEventListener("hashchange", renderRoute);
init();
