const DATA_ROOT = "/data";
const DATA_VERSION = "20260729-career-paths-v1";
const NEED_GAME_VALUES = [
  0, 5, 10, 15, 20, 25, 30, 35, 40,
  45, 50, 55, 60, 65, 70, 75, 80, 82,
];

const appState = {
  core: null,
  route: "overview",
  season: null,
  group: "forwards",
  glossaryGroup: "forwards",
  seasonTab: "snapshot",
  selectedPlayerId: null,
  seasonCache: new Map(),
  careers: null,
  careerGroup: "forwards",
  careerQuery: "",
  careerPlayerId: null,
  careerPlayerName: null,
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
  season: "Season level trends",
  career: "Career Trends",
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

function teamCodes(value) {
  return [...new Set(
    String(value || "")
      .split("/")
      .map((team) => team.trim().toUpperCase())
      .filter((team) => /^[A-Z0-9]{2,4}$/.test(team)),
  )];
}

function playerInitials(name) {
  return String(name || "")
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0] || "")
    .join("")
    .toUpperCase();
}

function assetSources(value) {
  return escapeHTML(JSON.stringify([...new Set(value.filter(Boolean))]));
}

function teamLogoSources(team, season) {
  const seasonKey = String(season || "");
  const historical = {
    ATL: "ATL_19992000-20102011_light.svg",
    PHX: "PHX_20032004-20132014_light.svg",
  };
  const filenames = [
    historical[team],
    team === "UTA" && seasonKey === "20242025"
      ? "UTA_20242025-20242025_light.svg"
      : null,
    `${team}_light.svg`,
    `${team}_dark.svg`,
  ].filter(Boolean);
  return filenames.map(
    (filename) => `https://assets.nhle.com/logos/nhl/svg/${filename}`,
  );
}

function playerHeadshotSources(player, season) {
  if (!player) return [];
  const teams = teamCodes(player.team);
  const seasonKey = /^\d{8}$/.test(String(season)) ? String(season) : "latest";
  const playerId = String(player.id || "").replace(/\D/g, "");
  if (!playerId) return [];
  return [
    ...teams.map(
      (team) =>
        `https://assets.nhle.com/mugs/nhl/${seasonKey}/${team}/${playerId}.png`,
    ),
    `https://assets.nhle.com/mugs/nhl/${seasonKey}/${playerId}.png`,
    ...teams.map(
      (team) =>
        `https://assets.nhle.com/mugs/nhl/latest/${team}/${playerId}.png`,
    ),
    `https://assets.nhle.com/mugs/nhl/latest/${playerId}.png`,
  ];
}

function playerVisual(player, season) {
  if (!player) return "";
  const teams = teamCodes(player.team);
  const seasonKey = /^\d{8}$/.test(String(season)) ? String(season) : "latest";
  const headshotSources = playerHeadshotSources(player, seasonKey);
  return `
    <div class="player-visual">
      <div class="player-headshot-frame" data-asset-frame>
        <span class="player-headshot-initials" aria-hidden="true">${escapeHTML(playerInitials(player.name))}</span>
        ${
          headshotSources.length
            ? `<img
                class="player-headshot"
                data-asset-sources="${assetSources(headshotSources)}"
                alt="${escapeHTML(player.name)} headshot"
                loading="lazy"
                decoding="async"
              />`
            : ""
        }
      </div>
      ${
        teams.length
          ? `
            <div class="team-logo-list" role="list" aria-label="${teams.length === 1 ? "Team" : "Teams"}">
              ${teams
                .map(
                  (team) => `
                    <span
                      class="team-logo-frame"
                      data-asset-frame
                      role="listitem"
                      aria-label="${escapeHTML(team)} team logo"
                      title="${escapeHTML(team)}"
                    >
                      <span class="team-logo-code" aria-hidden="true">${escapeHTML(team)}</span>
                      <img
                        class="team-logo"
                        data-asset-sources="${assetSources(teamLogoSources(team, seasonKey))}"
                        alt=""
                        loading="lazy"
                        decoding="async"
                      />
                    </span>
                  `,
                )
                .join("")}
            </div>
          `
          : ""
      }
    </div>
  `;
}

function rosterPlayerHeadshot(player, season) {
  const sources = playerHeadshotSources(player, season);
  return `
    <span class="roster-player-photo" data-asset-frame aria-hidden="true">
      <span class="roster-player-initials">${escapeHTML(playerInitials(player.name))}</span>
      ${
        sources.length
          ? `<img
              class="roster-player-photo-image"
              data-asset-sources="${assetSources(sources)}"
              alt=""
              loading="lazy"
              decoding="async"
            />`
          : ""
      }
    </span>
  `;
}

function rosterTeamLogo(team, season) {
  const sources = teamLogoSources(team, season);
  return `
    <span
      class="roster-team-logo"
      data-asset-frame
      role="img"
      aria-label="${escapeHTML(team)} team logo"
    >
      <span class="roster-team-logo-code" aria-hidden="true">${escapeHTML(team)}</span>
      <img
        class="roster-team-logo-image"
        data-asset-sources="${assetSources(sources)}"
        alt=""
        loading="lazy"
        decoding="async"
      />
    </span>
  `;
}

function playerIdentity(player, season, kicker, meta) {
  if (!player) return "";
  return `
    <div class="player-identity">
      ${playerVisual(player, season)}
      <div class="player-identity-copy">
        <p class="detail-kicker">${escapeHTML(kicker)}</p>
        <div class="detail-name">${escapeHTML(player.name)}</div>
        <p class="detail-meta">${escapeHTML(meta)}</p>
      </div>
    </div>
  `;
}

function hydratePlayerAssets(root = document) {
  root.querySelectorAll("img[data-asset-sources]").forEach((image) => {
    if (image.dataset.assetBound === "true") return;
    let sources = [];
    try {
      sources = JSON.parse(image.dataset.assetSources || "[]");
    } catch {
      sources = [];
    }
    const frame = image.closest("[data-asset-frame]");
    let sourceIndex = 0;
    const loadNext = () => {
      if (sourceIndex >= sources.length) {
        image.remove();
        frame?.classList.add("is-missing");
        return;
      }
      image.src = sources[sourceIndex];
      sourceIndex += 1;
    };
    image.dataset.assetBound = "true";
    image.addEventListener("load", () => {
      frame?.classList.remove("is-missing");
      frame?.classList.add("is-loaded");
    });
    image.addEventListener("error", loadNext);
    loadNext();
  });
}

function profileColor(name) {
  let hash = 0;
  for (const char of String(name)) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return PROFILE_COLORS[hash % PROFILE_COLORS.length];
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function profileMentionMarkup(value, profileNames) {
  const text = String(value ?? "");
  const names = [...new Set(
    (profileNames || [])
      .map((name) => String(name || "").trim())
      .filter(Boolean),
  )].sort((left, right) => right.length - left.length);
  if (!names.length) return escapeHTML(text);

  const expression = new RegExp(names.map(escapeRegExp).join("|"), "g");
  let markup = "";
  let offset = 0;
  for (const match of text.matchAll(expression)) {
    const index = match.index ?? offset;
    const name = match[0];
    markup += escapeHTML(text.slice(offset, index));
    markup += `<strong class="season-read-profile" style="--profile:${profileColor(name)}">${escapeHTML(name)}</strong>`;
    offset = index + name.length;
  }
  return `${markup}${escapeHTML(text.slice(offset))}`;
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

function minuteClock(value) {
  const totalSeconds = Math.max(0, Math.round(Number(value || 0) * 60));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function rosterValueBand(value) {
  const numeric = Number(value || 0);
  if (numeric >= 45) return "is-high";
  if (numeric >= 25) return "is-mid";
  return "is-low";
}

function confidenceBand(value) {
  const numeric = Number(value || 0);
  if (numeric >= 0.9) return "is-high";
  if (numeric >= 0.8) return "is-mid";
  return "is-low";
}

function xgBand(value) {
  if (value === null || value === undefined) return "is-neutral";
  const numeric = Number(value || 0);
  if (numeric >= 0.55) return "is-high";
  if (numeric >= 0.48) return "is-mid";
  return "is-low";
}

function needSimilarityBand(value) {
  const numeric = Number(value || 0);
  if (numeric >= 95) return "is-elite";
  if (numeric >= 90) return "is-strong";
  if (numeric >= 80) return "is-good";
  if (numeric >= 70) return "is-watch";
  if (numeric >= 55) return "is-light";
  return "is-low";
}

function needConfidenceBand(value) {
  const numeric = Number(value || 0) * 100;
  if (numeric > 90) return "is-high";
  if (numeric >= 80) return "is-mid";
  return "is-low";
}

function signedNumber(value) {
  const numeric = Number(value || 0);
  if (numeric > 0) return `+${number(numeric)}`;
  return number(numeric);
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
      getJSON(`${DATA_ROOT}/seasons/${season}.json?v=${DATA_VERSION}`),
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

function seasonMethodology() {
  return `
    <details class="methodology-expander">
      <summary>
        <span>A Quick Review of How Player Archetype is Calculated</span>
        <span class="methodology-toggle" aria-hidden="true"></span>
      </summary>
      <div class="methodology-body">
        <section class="methodology-section" aria-labelledby="method-data">
          <h2 id="method-data">Data used</h2>
          <p>
            I pulled <strong>public game-by-game NHL boxscore and time-on-ice data</strong>
            from the NHL Gamecenter endpoints, then aggregated it into
            <strong>regular season vs playoff</strong> splits.
          </p>
          <p>Each data point contributes to “style” like this:</p>
          <ul class="methodology-signals">
            <li><strong>Scoring/creation:</strong> shots, goals, assists, points → turned into per-60 rates (e.g., Shots/60)</li>
            <li><strong>Physical/defensive involvement:</strong> hits, blocks → per-60 rates</li>
            <li><strong>Puck pressure vs risk:</strong> takeaways vs giveaways → per-60 rates</li>
            <li><strong>Discipline/edge:</strong> penalty minutes → per-60 rate</li>
            <li><strong>Role/usage:</strong> PP TOI share and PK TOI share (how a coach deploys the player)</li>
            <li><strong>Deployment signals:</strong> faceoffs per game and faceoff percentage</li>
          </ul>
        </section>

        <section class="methodology-section" aria-labelledby="method-step-1">
          <h2 id="method-step-1">Step 1 — Normalize for ice time</h2>
          <p>Counting stats scale with ice time, so I convert them to <em>per-60</em> rates.</p>
          <div
            class="equation"
            role="math"
            aria-label="Shots per 60 equals shots divided by the quantity time on ice in seconds divided by 3,600"
          >
            <span>Shots/60</span>
            <span class="equation-symbol">=</span>
            <span class="fraction">
              <span>Shots</span>
              <span>TOI<sub>seconds</sub> / 3600</span>
            </span>
          </div>
          <p>I also compute special-teams usage share:</p>
          <div class="equation-row">
            <div
              class="equation"
              role="math"
              aria-label="Power-play share equals power-play time on ice divided by total time on ice"
            >
              <span>PP Share</span>
              <span class="equation-symbol">=</span>
              <span class="fraction">
                <span>PP TOI</span>
                <span>Total TOI</span>
              </span>
            </div>
            <div
              class="equation"
              role="math"
              aria-label="Penalty-kill share equals penalty-kill time on ice divided by total time on ice"
            >
              <span>PK Share</span>
              <span class="equation-symbol">=</span>
              <span class="fraction">
                <span>PK TOI</span>
                <span>Total TOI</span>
              </span>
            </div>
          </div>
        </section>

        <section class="methodology-section" aria-labelledby="method-step-2">
          <h2 id="method-step-2">Step 2 — Put every feature on the same scale</h2>
          <p>To keep extreme values from dominating the model, I robust-scale each feature:</p>
          <div
            class="equation"
            role="math"
            aria-label="x star equals the quantity x minus the median of x, divided by the interquartile range of x"
          >
            <span><var>x</var><sup>*</sup></span>
            <span class="equation-symbol">=</span>
            <span class="fraction">
              <span><var>x</var> − median(<var>x</var>)</span>
              <span>IQR(<var>x</var>)</span>
            </span>
          </div>
        </section>

        <section class="methodology-section" aria-labelledby="method-step-3">
          <h2 id="method-step-3">Step 3 — Compress the stats into a smaller “style fingerprint”</h2>
          <p>I reduce each skill block using Non-negative Matrix Factorization (NMF):</p>
          <div
            class="equation equation-compact"
            role="math"
            aria-label="X is approximately equal to W times H"
          >
            <var>X</var>
            <span class="equation-symbol">≈</span>
            <var>WH</var>
          </div>
          <p>
            You can think of each row of <strong>W</strong> as a compact
            <em>style fingerprint</em> for that player.
          </p>
        </section>

        <section class="methodology-section" aria-labelledby="method-step-4">
          <h2 id="method-step-4">Step 4 — Learn archetypes and assign probabilities</h2>
          <p>I fit a Gaussian Mixture Model (GMM) to those fingerprints:</p>
          <div
            class="equation equation-model"
            role="math"
            aria-label="p of z equals the sum from k equals 1 to K of pi k times a normal distribution of z given mu k and sigma k"
          >
            <span><var>p</var>(<var>z</var>)</span>
            <span class="equation-symbol">=</span>
            <span>∑<sub>k=1</sub><sup>K</sup> π<sub>k</sub> N(<var>z</var> | μ<sub>k</sub>, Σ<sub>k</sub>)</span>
          </div>
          <p>For each player (<var>i</var>), the model outputs a probability for each archetype:</p>
          <div
            class="equation equation-model"
            role="math"
            aria-label="p i k equals the probability that archetype equals k given z i"
          >
            <span><var>p</var><sub>ik</sub></span>
            <span class="equation-symbol">=</span>
            <span>P(Archetype = <var>k</var> | <var>z</var><sub>i</sub>)</span>
          </div>
        </section>

        <section
          class="methodology-section methodology-interpretation"
          aria-labelledby="method-blends"
        >
          <h2 id="method-blends">Why do blended style profiles exist?</h2>
          <p>
            The model is probabilistic: instead of forcing every player into exactly one
            bucket, it assigns a probability over archetypes. Some players genuinely
            combine traits that sit between multiple clusters (e.g., moderate scoring +
            moderate physical play), so their profile names describe the strongest trait
            combination rather than pretending every cluster is one clean role.
          </p>
          <p>
            <strong>Interpretation:</strong> if a player’s probabilities are
            (0.1, 87.3, 6.4, 6.3)% then the player is mostly aligned to one profile with
            <strong>87.3% confidence</strong>.
          </p>
        </section>
      </div>
    </details>
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

    const padding = {
      top: 16,
      right: width < 520 ? 28 : 16,
      bottom: 38,
      left: 44,
    };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const allValues = series.flatMap((item) => item.values.map((point) => point.value));
    const min = options.min ?? Math.floor(Math.min(...allValues) / 10) * 10;
    const max = options.max ?? Math.ceil(Math.max(...allValues) / 10) * 10;
    const range = Math.max(max - min, 1);
    const labels = series[0]?.values.map((point) => point.label) || [];

    context.font = "11px Inter, system-ui, sans-serif";
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
    const lastLabelIndex = labels.length - 1;
    labels.forEach((label, index) => {
      if (index !== lastLabelIndex && index % labelEvery !== 0) return;
      if (
        index !== lastLabelIndex &&
        lastLabelIndex - index < labelEvery * 0.8
      ) {
        return;
      }
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

function setupHeroRink(canvas) {
  if (!canvas) return;
  // NHL 2025-26 Official Rules, Section 1: 200' x 85' rink, 28' corners,
  // goal/blue-line locations, faceoff geometry, crease, and restricted area.
  const context = canvas.getContext("2d");
  const wrapper = canvas.parentElement;
  const styles = getComputedStyle(document.documentElement);
  const colors = {
    board: styles.getPropertyValue("--navy-900").trim(),
    ice: styles.getPropertyValue("--paper-strong").trim(),
    red: styles.getPropertyValue("--coral").trim(),
    aqua: styles.getPropertyValue("--aqua").trim(),
    gold: styles.getPropertyValue("--gold").trim(),
    blue: "#3974bb",
    crease: "rgba(33, 182, 168, 0.22)",
  };

  function draw() {
    const width = Math.max(wrapper.clientWidth, 250);
    const height = Math.max(wrapper.clientHeight, 250);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const padding = width < 320 ? 14 : 20;
    const scale = Math.min(
      (width - padding * 2) / 100,
      (height - padding * 2) / 85,
    );
    const rinkWidth = 100 * scale;
    const rinkHeight = 85 * scale;
    const left = (width - rinkWidth) / 2;
    const top = (height - rinkHeight) / 2;
    const x = (feet) => left + feet * scale;
    const y = (feet) => top + (42.5 + feet) * scale;
    const lineWidth = (feet, minimum = 1) =>
      Math.max(minimum, feet * scale);

    const rink = new Path2D();
    rink.moveTo(x(0), y(-42.5));
    rink.lineTo(x(72), y(-42.5));
    rink.arc(x(72), y(-14.5), 28 * scale, -Math.PI / 2, 0);
    rink.lineTo(x(100), y(14.5));
    rink.arc(x(72), y(14.5), 28 * scale, 0, Math.PI / 2);
    rink.lineTo(x(0), y(42.5));
    rink.closePath();

    context.save();
    context.clip(rink);
    context.fillStyle = colors.ice;
    context.fillRect(left, top, rinkWidth, rinkHeight);

    function drawVerticalLine(feet, color, thicknessFeet) {
      context.fillStyle = color;
      context.fillRect(
        x(feet - thicknessFeet / 2),
        y(-42.5),
        lineWidth(thicknessFeet),
        rinkHeight,
      );
    }

    drawVerticalLine(0, colors.red, 1);
    drawVerticalLine(25, colors.blue, 1);
    drawVerticalLine(89, colors.red, 2 / 12);

    function strokeCircle(cx, cy, radius, color, thicknessFeet = 2 / 12) {
      context.beginPath();
      context.arc(x(cx), y(cy), radius * scale, 0, Math.PI * 2);
      context.strokeStyle = color;
      context.lineWidth = lineWidth(thicknessFeet);
      context.stroke();
    }

    function drawSpot(cx, cy, radius, color) {
      context.beginPath();
      context.arc(x(cx), y(cy), radius * scale, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
    }

    strokeCircle(0, 0, 15, colors.blue);
    drawSpot(0, 0, 0.5, colors.blue);

    for (const spotY of [-22, 22]) {
      drawSpot(20, spotY, 1, colors.red);
      strokeCircle(69, spotY, 15, colors.red);
      drawSpot(69, spotY, 1, colors.red);

      context.strokeStyle = colors.red;
      context.lineWidth = lineWidth(2 / 12);
      for (const circleSide of [-1, 1]) {
        for (const offsetY of [-3.79, 3.79]) {
          context.beginPath();
          context.moveTo(x(69 + circleSide * 15), y(spotY + offsetY));
          context.lineTo(
            x(69 + circleSide * 17.25),
            y(spotY + offsetY),
          );
          context.stroke();
        }
      }

      for (const sideX of [-1, 1]) {
        for (const sideY of [-1, 1]) {
          const innerX = 69 + sideX * 2;
          const outerX = 69 + sideX * 6;
          const innerY = spotY + sideY * 2;
          const outerY = spotY + sideY * 5;
          context.beginPath();
          context.moveTo(x(innerX), y(innerY));
          context.lineTo(x(outerX), y(innerY));
          context.lineTo(x(outerX), y(outerY));
          context.stroke();
        }
      }
    }

    context.strokeStyle = colors.red;
    context.lineWidth = lineWidth(2 / 12);
    context.beginPath();
    context.moveTo(x(89), y(-11));
    context.lineTo(x(100), y(-14));
    context.moveTo(x(89), y(11));
    context.lineTo(x(100), y(14));
    context.stroke();

    const creaseRadius = 6 * scale;
    const creaseAngle = Math.atan2(4, Math.sqrt(20));
    context.beginPath();
    context.moveTo(x(89), y(-4));
    context.lineTo(x(89 - Math.sqrt(20)), y(-4));
    context.arc(
      x(89),
      y(0),
      creaseRadius,
      -Math.PI + creaseAngle,
      Math.PI - creaseAngle,
      true,
    );
    context.lineTo(x(89), y(4));
    context.closePath();
    context.fillStyle = colors.crease;
    context.fill();
    context.strokeStyle = colors.red;
    context.lineWidth = lineWidth(2 / 12);
    context.stroke();

    for (const creaseY of [-4, 4]) {
      context.beginPath();
      context.moveTo(x(85), y(creaseY));
      context.lineTo(x(85 + 5 / 12), y(creaseY));
      context.stroke();
    }

    context.strokeStyle = colors.red;
    context.lineWidth = Math.max(1.5, lineWidth(2 / 12));
    context.lineJoin = "round";
    context.beginPath();
    context.moveTo(x(89), y(-3));
    context.lineTo(x(92.68), y(-2.4));
    context.lineTo(x(92.68), y(2.4));
    context.lineTo(x(89), y(3));
    context.moveTo(x(89), y(-3));
    context.lineTo(x(89), y(3));
    context.stroke();

    function drawPlayArrow(start, controlOne, controlTwo, end, color) {
      context.save();
      context.beginPath();
      context.moveTo(x(start[0]), y(start[1]));
      context.bezierCurveTo(
        x(controlOne[0]),
        y(controlOne[1]),
        x(controlTwo[0]),
        y(controlTwo[1]),
        x(end[0]),
        y(end[1]),
      );
      context.strokeStyle = color;
      context.globalAlpha = 0.78;
      context.lineWidth = Math.max(1.5, lineWidth(7 / 12));
      context.lineCap = "round";
      context.stroke();

      const angle = Math.atan2(
        y(end[1]) - y(controlTwo[1]),
        x(end[0]) - x(controlTwo[0]),
      );
      const arrowSize = Math.max(5, 2.5 * scale);
      context.beginPath();
      context.moveTo(x(end[0]), y(end[1]));
      context.lineTo(
        x(end[0]) - Math.cos(angle - Math.PI / 6) * arrowSize,
        y(end[1]) - Math.sin(angle - Math.PI / 6) * arrowSize,
      );
      context.lineTo(
        x(end[0]) - Math.cos(angle + Math.PI / 6) * arrowSize,
        y(end[1]) - Math.sin(angle + Math.PI / 6) * arrowSize,
      );
      context.closePath();
      context.fillStyle = color;
      context.fill();
      context.restore();
    }

    function drawPlayMarker(type, markerX, markerY, color) {
      const markerSize = 2.35 * scale;
      context.save();
      context.strokeStyle = color;
      context.lineWidth = Math.max(1.8, lineWidth(3 / 4));
      context.lineCap = "round";
      if (type === "o") {
        context.beginPath();
        context.arc(x(markerX), y(markerY), markerSize, 0, Math.PI * 2);
        context.stroke();
      } else {
        context.beginPath();
        context.moveTo(x(markerX) - markerSize, y(markerY) - markerSize);
        context.lineTo(x(markerX) + markerSize, y(markerY) + markerSize);
        context.moveTo(x(markerX) + markerSize, y(markerY) - markerSize);
        context.lineTo(x(markerX) - markerSize, y(markerY) + markerSize);
        context.stroke();
      }
      context.restore();
    }

    drawPlayArrow([44, -21], [50, -31], [59, -29], [63, -17], colors.aqua);
    drawPlayArrow([56, 7], [64, -1], [72, -3], [78, 0], colors.gold);
    drawPlayArrow([51, 21], [59, 13], [68, 17], [74, 23], colors.aqua);

    [
      ["o", 42, -22, colors.board],
      ["o", 55, 8, colors.board],
      ["o", 77, 25, colors.board],
      ["x", 49, 23, colors.red],
      ["x", 65, -15, colors.red],
      ["x", 81, 2, colors.red],
    ].forEach(([type, markerX, markerY, color]) => {
      drawPlayMarker(type, markerX, markerY, color);
    });

    context.restore();
    context.strokeStyle = colors.board;
    context.lineWidth = Math.max(2, lineWidth(8 / 12));
    context.stroke(rink);
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
  const styleBreakdown = meta.namedStyleBreakdown;

  main.innerHTML = `
    <article class="page">
      <section class="hero">
        <div class="hero-copy">
          <p class="eyebrow">NHL PLAYER STYLE MODEL · <span class="season-token">${escapeHTML(oldest)} to ${escapeHTML(latest)}</span></p>
          <h1>See <em>how</em> players play, <span class="hero-accent"><em>beyond</em> the stats</span></h1>
          <div class="hero-actions">
            <a class="button button-primary" href="#season">Explore a season</a>
            <a class="button" href="#glossary">Browse styles</a>
          </div>
        </div>
        <div class="hero-board">
          <div class="rink-stage">
            <canvas
              id="hero-rink"
              role="img"
              aria-label="Regulation NHL half-rink with official markings, tactical X and O player positions, and curved movement arrows"
            ></canvas>
          </div>
          <div class="rink-facts" aria-label="${number(meta.playerCount)} players across ${meta.seasonCount} seasons">
            <span><strong>${number(meta.playerCount)}</strong> players</span>
            <span><strong>${meta.seasonCount}</strong> seasons</span>
          </div>
        </div>
      </section>

      <section class="metric-grid" aria-label="Dataset summary">
        <div class="metric metric-coverage">
          <span class="metric-label">Season coverage</span>
          <span class="metric-value metric-season-range">
            <span>${escapeHTML(oldest)}</span>
            <span><small>to</small> ${escapeHTML(latest)}</span>
          </span>
        </div>
        ${metric("Players analyzed", number(meta.playerCount), "NHL players")}
        ${metric("Different styles", number(meta.namedStyleCount), `${styleBreakdown.forwards} forward · ${styleBreakdown.defense} defense`)}
        ${metric("Avg. Model Confidence", `${number(meta.averageModelConfidence, 1)}%`, "across all seasons")}
      </section>

      <div class="overview-longform">
        <section class="story-intro">
          <p class="story-opening">In hockey, we often talk about a player's "identity" - enforcer or finisher or playmaker. It's one of hockey's most treasured features because a well-established identity is what often separates an NHL/AHL tweener from an NHL regular.</p>
          <p class="story-question">But, is identity truly a data-independent property or can we generate a data-driven approach to assigning identity?</p>
          <p>To answer the question above, I had to find a way to tackle the following:</p>
          <div class="question-grid">
            <p>What types of players exist in a given season?</p>
            <p>How is a roster constructed stylistically — and what’s missing?</p>
            <p>How does a player’s style evolve over the course of a career?</p>
          </div>
          <p>Everything here is generated from public NHL Gamecenter data combined with MoneyPuck player-level advanced metrics. The advanced-data era begins in 2008-09, which is the earliest season covered by the MoneyPuck files in this project.</p>
          <p>Use the left navigation to explore Play Style Glossary, Season Level Trends, Career Trends, and Playoff Trends.</p>
        </section>

        <section class="story-section" id="data-snapshot">
          <div class="story-heading">
            <h2>Big-Picture Snapshot of the Data</h2>
          </div>
          <div class="snapshot-grid">
            <div>
              <span>Seasons analyzed</span>
              <strong>${number(meta.seasonCount)}</strong>
            </div>
            <div>
              <span>Total NHL players</span>
              <strong>${number(meta.playerCount)}</strong>
            </div>
            <div>
              <span>Forward profiles</span>
              <strong>${number(meta.profileDefinitions.forwards)}</strong>
            </div>
            <div>
              <span>Defense profiles</span>
              <strong>${number(meta.profileDefinitions.defense)}</strong>
            </div>
          </div>
          <div class="definition-callout">
            <h3>What “style profile definitions” means:</h3>
            <p>Each season’s model learns K archetypes, then names them from the traits that are unusually high or low for that cluster. Some are clean roles like offense-driving, shot-blocking, or puck-pressure profiles; others are blended profiles whose names describe the strongest trait combination instead of using generic numbered labels.</p>
          </div>
          <div class="switch-grid">
            <div>
              <span>Forwards Median Switch Rate</span>
              <strong>${number(meta.switchRates.forwards, 2)}</strong>
            </div>
            <div>
              <span>Defense Median Switch Rate</span>
              <strong>${number(meta.switchRates.defense, 2)}</strong>
            </div>
          </div>
          <div class="definition-callout">
            <h3>Interpreting median switch rate:</h3>
            <p>Forwards median switch rate = 0.86 means the “typical” forward (among players with ≥3 seasons in the dataset) changes their top archetype in about 86% of year-to-year transitions.</p>
            <p>Defense median switch rate = 0.90 means the “typical” defenseman changes archetype in about 90% of transitions.</p>
          </div>
        </section>

        <section class="story-section" id="model-confidence">
          <div class="story-heading">
            <h2>How Confident was the Model Season by Season?</h2>
          </div>
          <div class="chart-panel overview-confidence-chart">
            <div class="legend" aria-label="Chart legend">
              <span><i style="--legend-color:var(--aqua)"></i>Forwards</span>
              <span><i style="--legend-color:var(--coral)"></i>Defense</span>
            </div>
            <div class="canvas-wrap">
              <canvas id="confidence-chart" role="img" aria-label="Average model confidence for forwards and defense by season"></canvas>
            </div>
          </div>
        </section>

        <section class="story-section methods-section" id="methods">
          <div class="story-heading">
            <h2>Methods</h2>
          </div>
          <p class="section-intro">At a high level, I’m learning a “style fingerprint” for each player-season using public data, then clustering those fingerprints into archetypes.</p>

          <div class="data-used">
            <h3>Data used</h3>
            <p>From NHL game endpoints I aggregate per player:</p>
            <ul>
              <li>regular season vs playoff statistics</li>
              <li>time on ice and special-teams usage</li>
              <li>boxscore counting stats (goals/assists/points/shots/hits/blocks/PIM/takeaways/giveaways, etc.)</li>
            </ul>
            <p>From MoneyPuck player game files I add advanced regular-season signals:</p>
            <ul>
              <li>expected goals, shot quality, high-danger chances, and rebound chances</li>
              <li>on-ice expected goals for/against and shot-attempt impact</li>
              <li>situation splits for 5-on-5, power play, and penalty kill</li>
              <li>shift starts, play-continuation metrics, penalties drawn, and faceoff context</li>
            </ul>
            <p>Because those MoneyPuck files start in 2008, the site focuses on seasons from 2008-09 forward.</p>
          </div>

          <div class="method-stack">
            <article class="method-detail">
              <span>01</span>
              <div>
                <h3>Step 1 — Normalize for ice time (so players are comparable)</h3>
                <p>Players have different ice time, so I convert raw counts into per-60 rates:</p>
                <div class="formula" aria-label="Shots per 60 equals shots divided by time on ice in seconds divided by 3600">
                  <span>Shots/60</span>
                  <b>=</b>
                  <span>Shots ÷ (TOI<sub>seconds</sub> / 3600)</span>
                </div>
                <p>Special-teams usage is represented as share of total TOI:</p>
                <div class="formula formula-pair">
                  <span>PP Share = PP TOI ÷ Total TOI</span>
                  <span>PK Share = PK TOI ÷ Total TOI</span>
                </div>
              </div>
            </article>

            <article class="method-detail">
              <span>02</span>
              <div>
                <h3>Step 2 — Put all features on the same scale</h3>
                <p>Some stats have heavy tails. To keep a few extreme values from dominating, I use a robust scaling transformation:</p>
                <div class="formula" aria-label="x star equals x minus median of x divided by the interquartile range of x">
                  <span>x<sup>*</sup></span>
                  <b>=</b>
                  <span>(x − median(x)) ÷ IQR(x)</span>
                </div>
              </div>
            </article>

            <article class="method-detail">
              <span>03</span>
              <div>
                <h3>Step 3 — Compress into a smaller “style fingerprint”</h3>
                <p>To summarize correlated features, I use Non-negative Matrix Factorization (NMF):</p>
                <div class="formula" aria-label="X approximately equals W H">
                  <span>X</span>
                  <b>≈</b>
                  <span>WH</span>
                </div>
                <p>Think of each row of W as a compact “style fingerprint” describing how a player produces their results.</p>
              </div>
            </article>

            <article class="method-detail">
              <span>04</span>
              <div>
                <h3>Step 4 — Learn archetypes with a probabilistic clustering model</h3>
                <p>I fit a Gaussian Mixture Model (GMM) to the fingerprints:</p>
                <div class="formula formula-wide" aria-label="p of z equals the sum from k equals 1 to K of pi k times a normal distribution">
                  <span>p(z)</span>
                  <b>=</b>
                  <span>Σ<sub>k=1</sub><sup>K</sup> π<sub>k</sub> N(z | μ<sub>k</sub>, Σ<sub>k</sub>)</span>
                </div>
                <p>For each player-season, the model outputs archetype probabilities using this formula:</p>
                <div class="formula formula-wide">
                  <span>p<sub>ik</sub></span>
                  <b>=</b>
                  <span>P(Archetype = k | z<sub>i</sub>)</span>
                </div>
                <p>Because this is soft clustering, a player can be “70% Playmaking Scorer, 20% Two-Way Creator, 10% Role Specialist” rather than being forced into a single bucket.</p>
                <p>I summarize how “mixed” a player is using:</p>
                <div class="formula formula-wide">
                  <span>Mixedness</span>
                  <b>=</b>
                  <span>1 − max<sub>k</sub>(p<sub>ik</sub>)</span>
                </div>
              </div>
            </article>
          </div>
        </section>

        <section class="story-section" id="references">
          <div class="story-heading">
            <h2>References</h2>
          </div>
          <p class="section-intro">This is a list of peer-reviewed papers, conference papers, and academic theses that I learned and took inspiration from while working on this project:</p>
          <ol class="reference-list">
            <li>
              <p>Gupta, P. (2025). Categorizing Playing Styles of Ice Hockey Players using Gaussian Mixture Models (GMM) and Non-negative Matrix Factorization (NMF).</p>
              <a href="https://liu.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&amp;c=23&amp;af=%5B%5D&amp;searchType=LIST_LATEST&amp;sortOrder2=title_sort_asc&amp;query=&amp;language=no&amp;pid=diva2%3A2004537&amp;aq=%5B%5B%5D%5D&amp;sf=all&amp;aqe=%5B%5D&amp;sortOrder=author_sort_asc&amp;onlyFullText=false&amp;noOfRows=50&amp;dswid=-733" target="_blank" rel="noreferrer">https://liu.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&amp;c=23&amp;af=%5B%5D&amp;searchType=LIST_LATEST&amp;sortOrder2=title_sort_asc&amp;query=&amp;language=no&amp;pid=diva2%3A2004537&amp;aq=%5B%5B%5D%5D&amp;sf=all&amp;aqe=%5B%5D&amp;sortOrder=author_sort_asc&amp;onlyFullText=false&amp;noOfRows=50&amp;dswid=-733</a>
            </li>
            <li>
              <p>Rosendahl, A. (2024). Player Type Classification in Ice Hockey Using Soft Clustering.</p>
              <a href="https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1886390&amp;dswid=-2788" target="_blank" rel="noreferrer">https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1886390</a>
            </li>
            <li>
              <p>Gupta, P. et al. (2025). A Gaussian Mixture Model Approach for Characterizing Playing Styles of Ice Hockey Players.</p>
              <a href="https://www.ida.liu.se/research/sportsanalytics/LINHAC/LINHAC25/papers/linhac25-paper7.pdf" target="_blank" rel="noreferrer">https://www.ida.liu.se/research/sportsanalytics/LINHAC/LINHAC25/papers/linhac25-paper7.pdf</a>
            </li>
            <li>
              <p>Schulte, O., Zhao, Z., Javan, M., Desaulniers, P. (2017). Apples-to-Apples: Clustering and Ranking NHL Players Using Location Information and Scoring Impact.</p>
              <a href="https://www.cs.sfu.ca/~oschulte/files/pubs/sloan-fix.pdf" target="_blank" rel="noreferrer">https://www.cs.sfu.ca/~oschulte/files/pubs/sloan-fix.pdf</a>
            </li>
            <li>
              <p>Macdonald, B. (2012). Adjusted Plus-Minus for NHL Players using Ridge Regression with Goals, Shots, Fenwick, and Corsi.</p>
              <a href="https://ideas.repec.org/a/bpj/jqsprt/v8y2012i3n8.html" target="_blank" rel="noreferrer">https://ideas.repec.org/a/bpj/jqsprt/v8y2012i3n8.html</a>
            </li>
          </ol>

          <div class="reference-subsection">
            <h3>Key open-source / reference links</h3>
            <ul class="link-list">
              <li>Zmalski — Unofficial NHL API Reference (api-web.nhle.com + stats/rest): <a href="https://github.com/Zmalski/NHL-API-Reference" target="_blank" rel="noreferrer">https://github.com/Zmalski/NHL-API-Reference</a></li>
              <li>Streamlit (app framework): <a href="https://streamlit.io" target="_blank" rel="noreferrer">https://streamlit.io</a></li>
              <li>st-aggrid (tables): <a href="https://github.com/PablocFonseca/streamlit-aggrid" target="_blank" rel="noreferrer">https://github.com/PablocFonseca/streamlit-aggrid</a></li>
            </ul>
          </div>
        </section>

        <section class="story-section" id="private-data">
          <div class="story-heading">
            <h2>What I'd do next with private tracking data</h2>
          </div>
          <p class="section-intro">The public NHL + MoneyPuck data can tell you a lot, but NHL teams have access to even richer behind-the-scenes streams. Here are the most natural extensions of this project and exactly what data I’d use if I had access to it.</p>

          <div class="extension-grid">
            <article>
              <span>01</span>
              <h3>1) Full-resolution puck &amp; player tracking</h3>
              <p>The NHL’s puck-and-player tracking system includes infrared cameras and emitters in pucks and sweaters that generates raw positional samples many times per second. While we do have Public NHL EDGE data, that data is curated and insulated from inspection by public. From conversations with NHL front office staff, teams have access to much more complete tracking data behind the scenes.</p>
              <h4>What I would look at + calculate:</h4>
              <ul>
                <li>skating acceleration bursts, high-speed entries, &amp; transition routes</li>
                <li>gap control and spacing (especially for defense)</li>
                <li>repeated sprint profiles &amp; fatigue signatures over shifts</li>
                <li>puck movement networks + “puck tempo” (how quickly the puck moves to dangerous space)</li>
              </ul>
            </article>

            <article>
              <span>02</span>
              <h3>2) Proprietary event data + video-linked analytics (i.e. Sportlogiq)</h3>
              <p>Vendors track far more than public play-by-play: pass types, forecheck pressure, retrievals, controlled exits/entries, lane creation, etc.</p>
              <h4>What I would look at + calculate:</h4>
              <ul>
                <li>true “puck pressure” and “play-driving” features from micro-events</li>
                <li>archetypes based on process (how plays are created) not just outcomes which is what I currently have here.</li>
              </ul>
            </article>

            <article>
              <span>03</span>
              <h3>3) Practice wearables &amp; sports science (i.e. Catapult)</h3>
              <p>Teams often collect practice load data from wearable sensors (IMUs) and sometimes heart-rate/physiology data.</p>
              <p>This data enables biomechanics-style insights (exertion vs. outcome asymmetry, fatigue, &amp; return-to-play baselines).</p>
              <h4>What I would look at + calculate:</h4>
              <ul>
                <li>workload-adjusted archetypes (how style changes under load)</li>
                <li>injury-risk / recovery-sensitive style shifts</li>
                <li>post-injury style drift detection</li>
              </ul>
            </article>

            <article>
              <span>04</span>
              <h3>4) Internal roster/contract/cap tools</h3>
              <p>Teams also have internal access to cap/contract information and transaction tools that the public can't access.</p>
              <h4>What I would look at + calculate:</h4>
              <ul>
                <li>link archetype needs to cap-efficient roster construction</li>
                <li>simulate “archetype coverage per dollar” as a roster optimization view</li>
              </ul>
            </article>
          </div>

          <div class="reference-subsection">
            <h3>Sources for the “behind the scenes” data streams:</h3>
            <ul class="link-list">
              <li>NHL EDGE and tracking system overview: <a href="https://www.nhl.com/news/nhl-edge-launches-website-for-puck-and-player-tracking-data" target="_blank" rel="noreferrer">https://www.nhl.com/news/nhl-edge-launches-website-for-puck-and-player-tracking-data</a></li>
              <li>Wearable practice tech + examples (Catapult etc): <a href="https://www.dailyfaceoff.com/news/nhlpa-reminds-players-of-their-right-to-control-or-destroy-wearable-tech-data" target="_blank" rel="noreferrer">https://www.dailyfaceoff.com/news/nhlpa-reminds-players-of-their-right-to-control-or-destroy-wearable-tech-data</a></li>
              <li>Catapult hockey wearables overview: <a href="https://www.catapult.com/sports/ice-hockey" target="_blank" rel="noreferrer">https://www.catapult.com/sports/ice-hockey</a></li>
              <li>Example of wearable tracking described publicly: <a href="https://www.si.com/edge/2015/02/20/tech-talk-catapult-tracking-nhl-data-injury-reduction" target="_blank" rel="noreferrer">https://www.si.com/edge/2015/02/20/tech-talk-catapult-tracking-nhl-data-injury-reduction</a></li>
              <li>Sportlogiq hockey platform: <a href="https://www.sportlogiq.com/hockey/" target="_blank" rel="noreferrer">https://www.sportlogiq.com/hockey/</a></li>
              <li>NHL cap/contract iPad app (team-side tooling): <a href="https://apnews.com/article/49b2f0421df504555bbc940bb861e4a2" target="_blank" rel="noreferrer">https://apnews.com/article/49b2f0421df504555bbc940bb861e4a2</a></li>
            </ul>
          </div>
        </section>
        <p class="data-disclaimer">I try to keep the data as up-to-date as possible but there's always a chance that things may be out of date.</p>
      </div>
    </article>
  `;

  setupHeroRink(document.querySelector("#hero-rink"));
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
  return appState.core.glossary[appState.glossaryGroup];
}

function glossaryExampleLink(example) {
  const isRecord = example && typeof example === "object";
  const name = String(isRecord ? example.name || "" : example || "").trim();
  if (!name) return "";
  const playerId = Number(isRecord ? example.id : 0);
  const destination =
    Number.isInteger(playerId) && playerId > 0
      ? `#career?player=${playerId}&amp;group=${escapeHTML(appState.glossaryGroup)}`
      : `#career?name=${encodeURIComponent(name)}&amp;group=${escapeHTML(appState.glossaryGroup)}`;
  return `
    <a
      href="${destination}"
      target="_blank"
      rel="noopener noreferrer"
      aria-label="Open ${escapeHTML(name)}’s profile in a new tab"
    >${escapeHTML(name)}</a>
  `;
}

function updateGlossaryList() {
  const rows = glossaryRows();
  const list = document.querySelector("#glossary-list");
  if (!list) return;
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
                <div class="example-links">
                  ${
                    row.examples
                      .map(glossaryExampleLink)
                      .filter(Boolean)
                      .join("") || "<span>No recent examples</span>"
                  }
                </div>
              </div>
            </article>
          `,
        )
        .join("")
    : `<div class="empty-state">No styles are available.</div>`;
}

function renderGlossary() {
  main.innerHTML = `
    <article class="page">
      ${pageHeader(
        "Style glossary",
        "What Are All the Player Styles According to the Model?",
        "This glossary aggregates the named archetypes learned across all available seasons. Each row shows the clearest statistical signature and a few recent example players.",
        groupControl(appState.glossaryGroup, "glossary-group"),
      )}
      <section class="glossary-list" id="glossary-list" aria-live="polite"></section>
      <footer class="page-footer">
        <span>Profiles are learned independently each season.</span>
        <span>Examples link to each player’s career profile.</span>
      </footer>
    </article>
  `;

  bindGroupControl("glossary-group", (group) => {
    appState.glossaryGroup = group;
    renderGlossary();
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
  return `
    <div class="bar-list">
      ${profiles
        .slice(0, limit)
        .map(
          (profile) => {
            const share = Math.min(
              100,
              Math.max(0, Number(profile.share) || 0),
            );
            return `
            <div class="bar-row">
              <span class="bar-label">${escapeHTML(profile.name)}</span>
              <span class="bar-track" aria-hidden="true">
                <span
                  class="bar-fill"
                  style="width:${share}%;--profile:${profileColor(profile.name)}"
                ></span>
              </span>
              <span class="bar-value">${number(profile.share, 1)}%</span>
            </div>
          `;
          },
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
  const profileNames = [
    ...(appState.core?.glossary?.[appState.group] || []).map(
      (profile) => profile.name,
    ),
    ...groupData.profiles.map((profile) => profile.name),
  ];
  const seasonRead = groupData.seasonRead || {
    headline: `${dominant.name} leads the season`,
    paragraphs: [
      `${dominant.name} accounts for ${number(dominant.share, 1)}% of the group, while the three most common styles cover ${number(topThree, 1)}%.`,
    ],
    facts: [
      {
        label: "Average confidence",
        value: percent(groupData.averageConfidence, 1),
      },
      {
        label: "Styles present",
        value: number(groupData.profiles.length),
      },
      {
        label: "Mixed profiles",
        value: number(groupData.mixedCount),
      },
    ],
  };
  return `
    <section class="metric-grid">
      ${metric("Players", number(groupData.players.length))}
      ${metric("Dominant style", `${number(dominant.share, 1)}%`, dominant.name)}
      ${metric(
        "Top-three share",
        `${number(topThree, 1)}%`,
        "Players in that season’s 3 most common learned styles",
      )}
      ${metric("Mixed profiles", number(groupData.mixedCount), "below 80% top-style confidence")}
    </section>
    <section class="analysis-grid season-snapshot-layout">
      <div class="chart-panel">
        <div class="chart-title-row">
          <div>
            <h3>Roster-wide style mix</h3>
            <p>Share of player-seasons in each top profile.</p>
          </div>
        </div>
        ${profileBars(groupData.profiles)}
      </div>
      <article class="info-panel season-read-panel">
        <div class="season-read-heading">
          <p class="eyebrow">Season read</p>
          <h3>${escapeHTML(seasonRead.headline)}</h3>
        </div>
        <div class="season-read-copy">
          ${seasonRead.paragraphs
            .map(
              (paragraph) =>
                `<p>${profileMentionMarkup(paragraph, profileNames)}</p>`,
            )
            .join("")}
        </div>
        <div class="season-read-facts" aria-label="Season read supporting figures">
          ${seasonRead.facts
            .map(
              (fact) => `
                <div class="season-read-fact">
                  <strong>${escapeHTML(fact.value)}</strong>
                  <span>${escapeHTML(fact.label)}</span>
                </div>
              `,
            )
            .join("")}
        </div>
      </article>
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
      ${playerIdentity(
        player,
        appState.season,
        "Player profile",
        [player.position, player.team].filter(Boolean).join(" · "),
      )}
      ${profileChip(player.profile)}
      <div class="detail-stats">
        <div class="detail-stat"><span>Games played</span><strong>${number(player.games)}</strong></div>
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

function rosterConcentrationRing(label, value, subtitle, profile) {
  const bounded = Math.max(0, Math.min(100, Number(value || 0)));
  return `
    <article class="roster-ring-card">
      <div
        class="roster-ring"
        style="--ring:${profileColor(profile)};--ring-value:${bounded}"
        role="img"
        aria-label="${escapeHTML(label)}: ${number(bounded)} percent ${escapeHTML(profile)}"
      >
        <div class="roster-ring-core">
          <strong>${number(bounded)}%</strong>
          <span>${escapeHTML(label)}</span>
        </div>
      </div>
      <p>${escapeHTML(subtitle)}</p>
    </article>
  `;
}

function rosterUnitCard(unit, construction, season) {
  const xgText =
    unit.xgPct === null || unit.xgPct === undefined
      ? "xG n/a*"
      : `${percent(unit.xgPct)} xG`;
  return `
    <article
      class="roster-unit-card"
      style="--unit-profile:${profileColor(unit.profile)}"
    >
      <header class="roster-unit-head">
        <strong>${escapeHTML(construction.unitLabel)} ${number(unit.number)}</strong>
        <span class="roster-unit-meta">
          ${number(unit.minutes)} min ·
          <span class="roster-status-chip ${xgBand(unit.xgPct)}">${escapeHTML(xgText)}</span>
        </span>
      </header>
      <div class="roster-unit-players">
        ${unit.players
          .map(
            (player) => `
              <article
                class="roster-skater"
                style="--profile:${profileColor(player.profile)}"
              >
                ${rosterPlayerHeadshot(player, season)}
                <div class="roster-skater-copy">
                  <div class="roster-skater-name">
                    <strong>${escapeHTML(player.name)}</strong>
                    <span class="roster-status-chip ${confidenceBand(player.confidence)}">
                      ${number(Number(player.confidence || 0) * 100, 1)}%
                    </span>
                  </div>
                  <p>
                    ${escapeHTML(player.position)} · ${number(player.games)} GP ·
                    ${minuteClock(player.atoi)} ATOI · ${number(player.goals)}G,
                    ${number(player.assists)}A, ${number(player.points)}P
                  </p>
                  <span class="roster-profile-label">${escapeHTML(player.profile)}</span>
                </div>
              </article>
            `,
          )
          .join("")}
      </div>
    </article>
  `;
}

function rosterMixTable(construction) {
  return `
    <div class="table-wrap roster-table-wrap">
      <table class="data-table roster-analysis-table">
        <thead>
          <tr>
            <th>Archetype</th>
            <th class="numeric">Overall (%)</th>
            <th class="numeric">${escapeHTML(construction.topLabel)} (%)</th>
            <th class="numeric">${escapeHTML(construction.bottomLabel)} (%)</th>
          </tr>
        </thead>
        <tbody>
          ${construction.mix
            .map(
              (row) => `
                <tr>
                  <td>${profileChip(row.profile)}</td>
                  <td class="numeric roster-value ${rosterValueBand(row.overall)}">${number(row.overall, 1)}</td>
                  <td class="numeric roster-value ${rosterValueBand(row.top)}">${number(row.top, 1)}</td>
                  <td class="numeric roster-value ${rosterValueBand(row.bottom)}">${number(row.bottom, 1)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function rosterDepthTable(construction) {
  const players = construction.units.flatMap((unit) =>
    unit.players.map((player) => ({
      ...player,
      unit: unit.number,
    })),
  );
  return `
    <div class="table-wrap roster-table-wrap">
      <table class="data-table roster-depth-table">
        <thead>
          <tr>
            <th>${escapeHTML(construction.unitLabel)}</th>
            <th>Depth</th>
            <th>Player</th>
            <th>Pos</th>
            <th>Archetype</th>
            <th class="numeric">REG GP</th>
            <th class="numeric">REG ATOI</th>
            <th class="numeric">REG P</th>
            <th class="numeric">REG G</th>
            <th class="numeric">REG A</th>
            <th class="numeric">Confidence</th>
          </tr>
        </thead>
        <tbody>
          ${players
            .map(
              (player) => `
                <tr>
                  <td>${number(player.unit)}</td>
                  <td>${number(player.depth)}</td>
                  <td><strong>${escapeHTML(player.name)}</strong></td>
                  <td>${escapeHTML(player.position)}</td>
                  <td>${profileChip(player.profile)}</td>
                  <td class="numeric">${number(player.games)}</td>
                  <td class="numeric">${minuteClock(player.atoi)}</td>
                  <td class="numeric">${number(player.points)}</td>
                  <td class="numeric">${number(player.goals)}</td>
                  <td class="numeric">${number(player.assists)}</td>
                  <td class="numeric">${number(Number(player.confidence || 0) * 100, 1)}%</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function rosterGapTable(construction) {
  return `
    <div class="table-wrap roster-table-wrap">
      <table class="data-table roster-gap-table">
        <thead>
          <tr>
            <th>Archetype</th>
            <th class="numeric">Team share (%)</th>
            <th class="numeric">League avg (%)</th>
            <th class="numeric">Z-score</th>
            <th class="numeric">Strong coverage (%)</th>
            <th class="numeric">Reliance on top 2 (%)</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          ${construction.gaps
            .map(
              (row) => `
                <tr>
                  <td>${profileChip(row.profile)}</td>
                  <td class="numeric">${number(row.teamShare, 1)}</td>
                  <td class="numeric">${number(row.leagueAverage, 1)}</td>
                  <td class="numeric">${number(row.zScore, 2)}</td>
                  <td class="numeric">${number(row.strongCoverage, 1)}</td>
                  <td class="numeric">${number(row.topTwoReliance, 1)}</td>
                  <td>${row.note ? `<span class="roster-gap-note">${escapeHTML(row.note)}</span>` : "—"}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function teamView(construction, season, group) {
  if (!construction) {
    return '<div class="empty-state">No roster construction data is available for that team.</div>';
  }
  const dominant = construction.dominant;
  const rosterCount = construction.units.reduce(
    (total, unit) => total + unit.players.length,
    0,
  );
  const groupLabel = group === "forwards" ? "Forwards" : "Defense";
  const sourceCaption = `Units are selected from ${construction.source}.`;
  return `
    <section class="team-construction">
      <header class="roster-team-header">
        ${rosterTeamLogo(construction.team, season)}
        <div>
          <p class="eyebrow">Team roster construction</p>
          <h2>${escapeHTML(construction.team)}</h2>
          <p>${escapeHTML(groupLabel)} · ${number(rosterCount)}-player depth view</p>
        </div>
      </header>

      <section class="roster-ring-grid" aria-label="Style concentration">
        ${rosterConcentrationRing(
          "Overall",
          dominant.overall,
          dominant.profile,
          dominant.profile,
        )}
        ${rosterConcentrationRing(
          construction.topLabel,
          dominant.top,
          `${dominant.profile} concentration`,
          dominant.profile,
        )}
        ${rosterConcentrationRing(
          construction.bottomLabel,
          dominant.bottom,
          `${dominant.profile} concentration`,
          dominant.profile,
        )}
      </section>

      <section class="roster-summary-grid">
        <article>
          <span>Dominant profile</span>
          <strong>${escapeHTML(dominant.profile)}</strong>
        </article>
        <article>
          <span>Top/bottom gap</span>
          <strong>${number(dominant.gap)} pts</strong>
        </article>
      </section>

      <section class="roster-units-section">
        <div class="section-heading roster-section-heading">
          <div>
            <h3>${escapeHTML(construction.team)} ${escapeHTML(construction.unitLabel.toLowerCase())} construction</h3>
            <p>${escapeHTML(sourceCaption)}</p>
          </div>
        </div>
        <div class="roster-unit-grid">
          ${construction.units
            .map((unit) => rosterUnitCard(unit, construction, season))
            .join("")}
        </div>
        ${
          construction.hasFallbackUnits
            ? '<p class="roster-footnote">* xG data is unavailable for fallback units; their minutes are the sum of regular-season player TOI.</p>'
            : ""
        }
      </section>

      <section class="roster-analysis-section">
        <div class="section-heading roster-section-heading">
          <div><h3>Roster profile mix</h3></div>
        </div>
        ${rosterMixTable(construction)}
      </section>

      <section class="roster-analysis-section">
        <div class="section-heading roster-section-heading">
          <div><h3>Depth chart table</h3></div>
        </div>
        ${rosterDepthTable(construction)}
      </section>

      <section class="roster-analysis-section roster-gap-section">
        <div class="section-heading roster-section-heading">
          <div>
            <h3>League-context roster gaps</h3>
            <p>Gap score compares this team's profile mix to the rest of the league using the same selected group.</p>
          </div>
        </div>
        ${rosterGapTable(construction)}
      </section>
    </section>
  `;
}

function renderSeasonTeams(groupData, selectedTeam = "") {
  const constructions = groupData.teamConstructions || {};
  const teams = Object.keys(constructions).sort();
  const team = teams.includes(selectedTeam) ? selectedTeam : teams[0] || "";
  return `
    <section class="roster-intro info-panel">
      <h2>Team Roster Construction</h2>
      <div class="roster-intro-grid">
        <div>
          <h3>What you’re looking at</h3>
          <ul>
            <li>A depth-chart view of the selected team using the 12 forwards or 8 defensemen with the most regular-season ice time.</li>
            <li>Style concentration rings showing the dominant archetype overall and how much of it lives in the top half vs bottom half of the roster.</li>
          </ul>
        </div>
        <div>
          <h3>What you can learn</h3>
          <ul>
            <li>Whether the team identity is concentrated in stars or spread through depth.</li>
            <li>Which lines/pairs carry each profile.</li>
            <li>Where the roster construction is balanced or thin.</li>
          </ul>
        </div>
      </div>
    </section>
    <div class="roster-team-control">
      <div class="field">
        <label for="team-select">Team</label>
        <select id="team-select">
          ${teams
            .map(
              (value) =>
                `<option value="${escapeHTML(value)}"${value === team ? " selected" : ""}>${escapeHTML(value)}</option>`,
            )
            .join("")}
        </select>
      </div>
    </div>
    <div id="team-view">${teamView(constructions[team], appState.season, appState.group)}</div>
  `;
}

function Counter(values) {
  const counts = new Map();
  values.forEach((value) => counts.set(value, (counts.get(value) || 0) + 1));
  return counts;
}

function needFinderTargets(groupData) {
  const configured = groupData.needFinder?.targets || [];
  if (configured.length) return configured;
  return groupData.profiles.map((profile, cluster) => ({
    profile: profile.name,
    cluster,
  }));
}

function needTeamMarks(player, season) {
  const teams = teamCodes(player.team);
  if (!teams.length) return "";
  return `
    <span class="need-team-list" role="list" aria-label="${teams.length === 1 ? "Team" : "Teams"}">
      ${teams
        .map(
          (team) => `
            <span
              class="need-team-mark"
              data-asset-frame
              role="listitem"
              aria-label="${escapeHTML(team)}"
              title="${escapeHTML(team)}"
            >
              <span class="need-team-code" aria-hidden="true">${escapeHTML(team)}</span>
              <img
                class="need-team-logo"
                data-asset-sources="${assetSources(teamLogoSources(team, season))}"
                alt=""
                loading="lazy"
                decoding="async"
              />
            </span>
          `,
        )
        .join("")}
      <span class="need-team-text">${escapeHTML(player.team)}</span>
    </span>
  `;
}

function needStat(label, value) {
  return `
    <div class="need-stat">
      <dt>${escapeHTML(label)}</dt>
      <dd>${escapeHTML(value)}</dd>
    </div>
  `;
}

function playerTargetSimilarity(player, cluster, targetProfile) {
  const direct = Number(player.targetScores?.[cluster]);
  if (Number.isFinite(direct)) return direct;
  const fallback = player.probabilities.find(
    (item) => item.profile === targetProfile,
  );
  return Number(((fallback?.value || 0) * 100).toFixed(1));
}

function needGamesValue(control) {
  const index = Math.max(
    0,
    Math.min(NEED_GAME_VALUES.length - 1, Number(control?.value || 0)),
  );
  return NEED_GAME_VALUES[index];
}

function needPlayerCard(player, rank, targetSimilarity, details) {
  const similarityBand = needSimilarityBand(targetSimilarity);
  const confidenceClass = needConfidenceBand(player.confidence);
  const careerHref = `#career?player=${encodeURIComponent(player.id)}&amp;group=${escapeHTML(appState.group)}`;
  const styleDetails = details?.[String(player.cluster)] || {
    name: player.profile,
    summary: "",
    higher: "None",
    lower: "None",
  };
  const progress = Math.max(0, Math.min(100, targetSimilarity));
  return `
    <details class="need-player-card" style="--profile:${profileColor(player.profile)}">
      <summary class="need-player-summary">
        <span class="need-rank" aria-label="Rank ${rank}">${String(rank).padStart(2, "0")}</span>
        ${rosterPlayerHeadshot(player, appState.season)}
        <span class="need-player-identity">
          <strong>${escapeHTML(player.name)}</strong>
          <span>${escapeHTML(player.position)}</span>
          ${needTeamMarks(player, appState.season)}
        </span>
        <span class="need-similarity ${similarityBand}">
          <span class="need-summary-label">Target similarity</span>
          <strong>${number(targetSimilarity, 1)}%</strong>
          <span
            class="need-similarity-track"
            role="progressbar"
            aria-label="${escapeHTML(player.name)} target similarity"
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow="${escapeHTML(targetSimilarity)}"
          >
            <span style="width:${progress}%"></span>
          </span>
        </span>
        <span class="need-current-style">
          <span class="need-summary-label">Archetype</span>
          ${profileChip(player.profile)}
          <span class="need-confidence ${confidenceClass}">
            ${number(Number(player.confidence || 0) * 100, 1)}% confidence
          </span>
        </span>
        <span class="need-quick-stats" aria-label="Regular-season summary">
          <span><small>GP</small><strong>${number(player.games)}</strong></span>
          <span><small>PTS</small><strong>${number(player.points)}</strong></span>
          <span><small>ATOI</small><strong>${escapeHTML(player.regAtoi || minuteClock(player.toi))}</strong></span>
        </span>
        <span class="need-disclosure" aria-hidden="true"></span>
      </summary>
      <div class="need-player-detail">
        <section class="need-style-detail">
          <p class="eyebrow">Archetype details</p>
          <h4>${escapeHTML(styleDetails.name || player.profile)}</h4>
          ${styleDetails.summary ? `<p>${escapeHTML(styleDetails.summary)}</p>` : ""}
          <dl class="need-trait-list">
            <div>
              <dt>Higher traits</dt>
              <dd>${escapeHTML(styleDetails.higher || "None")}</dd>
            </div>
            <div>
              <dt>Lower traits</dt>
              <dd>${escapeHTML(styleDetails.lower || "None")}</dd>
            </div>
          </dl>
          <a class="need-career-link" href="${careerHref}">Open player profile</a>
        </section>
        <section class="need-season-stats">
          <h4>Regular season</h4>
          <dl class="need-stat-grid">
            ${needStat("GP", number(player.games))}
            ${needStat("ATOI", player.regAtoi || minuteClock(player.toi))}
            ${needStat("P", number(player.points))}
            ${needStat("G", number(player.goals))}
            ${needStat("A", number(player.assists))}
            ${needStat("SOG", number(player.shots))}
            ${needStat("+/-", signedNumber(player.plusMinus))}
            ${needStat("PIM", number(player.pim))}
          </dl>
        </section>
        <section class="need-season-stats">
          <h4>Playoffs</h4>
          <dl class="need-stat-grid">
            ${needStat("GP", number(player.playoffGames))}
            ${needStat("ATOI", player.playoffAtoi || minuteClock(player.playoffToi))}
            ${needStat("P", number(player.playoffPoints))}
            ${needStat("G", number(player.playoffGoals))}
            ${needStat("A", number(player.playoffAssists))}
            ${needStat("SOG", number(player.playoffShots))}
            ${needStat("+/-", signedNumber(player.playoffPlusMinus))}
            ${needStat("PIM", number(player.playoffPim))}
          </dl>
        </section>
      </div>
    </details>
  `;
}

function renderNeedFinder(groupData) {
  const targets = needFinderTargets(groupData);
  const teams = [...new Set(
    groupData.players.flatMap((player) => player.team.split("/").filter(Boolean)),
  )].sort();
  const target = targets[0] || { profile: "", cluster: 0 };
  return `
    <section class="need-finder-intro">
      <div class="need-finder-title">
        <h2>Need Finder (find players who match a target archetype)</h2>
      </div>
      <div class="need-finder-guide">
        <div>
          <h3>What you’re looking at</h3>
          <ul>
            <li>A ranked list of players who best match a selected style profile.</li>
          </ul>
        </div>
        <div>
          <h3>How to use it</h3>
          <ul>
            <li>Pick the archetype you want to add to a roster.</li>
            <li>Optionally exclude your own team.</li>
            <li>Increase minimum regular-season games to avoid tiny samples.</li>
            <li>“Target similarity (%)” is the model’s estimated probability that the player belongs to that archetype.</li>
          </ul>
        </div>
      </div>
    </section>
    <section class="need-finder-controls" aria-label="Need finder filters">
      <div class="field">
        <label for="need-team">Exclude team (optional)</label>
        <select id="need-team"><option value="">(none)</option>${teams.map((team) => `<option value="${escapeHTML(team)}">${escapeHTML(team)}</option>`).join("")}</select>
      </div>
      <div class="field">
        <label for="need-profile">Target archetype</label>
        <select id="need-profile">${targets
          .map(
            (item) =>
              `<option value="${escapeHTML(item.cluster)}">${escapeHTML(item.profile)}</option>`,
          )
          .join("")}</select>
      </div>
      <div class="field need-games-field">
        <div class="need-range-label">
          <label for="need-games">Min REG games</label>
          <output id="need-games-value" for="need-games">20</output>
        </div>
        <input
          id="need-games"
          type="range"
          min="0"
          max="${NEED_GAME_VALUES.length - 1}"
          value="${NEED_GAME_VALUES.indexOf(20)}"
          step="1"
          aria-valuemin="0"
          aria-valuemax="82"
          aria-valuenow="20"
          aria-valuetext="20 regular-season games"
        />
        <div class="need-range-ends" aria-hidden="true"><span>0</span><span>82</span></div>
      </div>
    </section>
    <div id="need-results" aria-live="polite">${needResults(groupData, target.cluster, target.profile, "", 20)}</div>
  `;
}

function needResults(groupData, targetCluster, targetProfile, excludedTeam, minGames) {
  const eligible = groupData.players
    .map((player) => ({
      ...player,
      targetSimilarity: playerTargetSimilarity(
        player,
        targetCluster,
        targetProfile,
      ),
    }))
    .filter(
      (player) =>
        player.games >= minGames &&
        (!excludedTeam || !String(player.team || "").includes(excludedTeam)),
    )
    .sort(
      (a, b) =>
        b.targetSimilarity - a.targetSimilarity ||
        Number(b.points || 0) - Number(a.points || 0) ||
        Number(a.needOrder || 0) - Number(b.needOrder || 0),
    );
  const matches = eligible.slice(0, 80);
  if (!matches.length) {
    return `<div class="empty-state">No players meet those filters.</div>`;
  }
  return `
    <section class="need-results-board">
      <header class="need-results-header">
        <div>
          <p class="eyebrow">Ranked matches</p>
          <h3>${escapeHTML(targetProfile)}</h3>
          <p>Ordered by target similarity, then regular-season points.</p>
        </div>
        <div class="need-results-count">
          <strong>${number(matches.length)}</strong>
          <span>of ${number(eligible.length)} eligible players shown</span>
        </div>
      </header>
      <ol class="need-result-list">
        ${matches
          .map(
            (player, index) => `
              <li>
                ${needPlayerCard(
                  player,
                  index + 1,
                  player.targetSimilarity,
                  groupData.needFinder?.details,
                )}
              </li>
            `,
          )
          .join("")}
      </ol>
    </section>
  `;
}

async function renderSeason() {
  const seasonLabel =
    appState.core.meta.seasons.find((item) => item.key === appState.season)?.label ||
    appState.season;
  main.innerHTML = `
    <article class="page season-page">
      ${pageHeader(
        "Season level trends",
        "What Are the Season Level Trends in Play Style?",
        "Explore league-wide styles, players, teams, and roster needs for any season.",
        seasonControls(),
      )}
      ${seasonMethodology()}
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
        ["teams", "Team roster construction"],
        ["needs", "Need Finder"],
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
        const teamViewElement = document.querySelector("#team-view");
        teamViewElement.innerHTML = teamView(
          groupData.teamConstructions?.[event.target.value],
          appState.season,
          appState.group,
        );
        hydratePlayerAssets(teamViewElement);
      });
    } else {
      panel.innerHTML = renderNeedFinder(groupData);
      bindNeedFinder(groupData);
    }
    hydratePlayerAssets(panel);
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
    hydratePlayerAssets(detail);
    drawRows();
  });
}

function bindNeedFinder(groupData) {
  const profile = document.querySelector("#need-profile");
  const team = document.querySelector("#need-team");
  const games = document.querySelector("#need-games");
  const results = document.querySelector("#need-results");
  const output = document.querySelector("#need-games-value");
  const targets = needFinderTargets(groupData);
  const draw = () => {
    if (!profile || !team || !games || !results) return;
    const targetCluster = Number(profile.value);
    const target = targets.find(
      (item) => Number(item.cluster) === targetCluster,
    );
    const minGames = needGamesValue(games);
    if (output) output.textContent = minGames;
    games.setAttribute("aria-valuenow", String(minGames));
    games.setAttribute(
      "aria-valuetext",
      `${minGames} regular-season games`,
    );
    results.innerHTML = needResults(
      groupData,
      targetCluster,
      target?.profile || profile.options[profile.selectedIndex]?.text || "",
      team.value,
      minGames,
    );
    hydratePlayerAssets(results);
  };
  profile?.addEventListener("change", draw);
  team?.addEventListener("change", draw);
  games?.addEventListener("input", draw);
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
          seasons: 0,
          firstSeason: record.season,
          lastSeason: record.season,
          latestSeason: record.season,
          position: record.position,
          team: record.team,
        });
      }
      const player = grouped.get(record.id);
      player.seasons += 1;
      if (record.season < player.firstSeason) player.firstSeason = record.season;
      if (record.season > player.lastSeason) player.lastSeason = record.season;
      if (record.season >= player.latestSeason) {
        player.latestSeason = record.season;
        player.position = record.position;
        player.team = record.team;
      }
    });
  return [...grouped.values()]
    .map((player) => ({
      ...player,
      display: `${player.name} — ${player.position || "UNK"} — ${careerSeasonLabel(player.firstSeason)} - ${careerSeasonLabel(player.lastSeason)}`,
    }))
    .sort(
      (a, b) =>
        b.latestSeason.localeCompare(a.latestSeason) ||
        a.name.localeCompare(b.name),
    );
}

function careerSearchResults(players, query) {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return players;
  return players.filter((player) =>
    player.name.toLowerCase().includes(normalized),
  );
}

function careerSeasonLabel(season) {
  const key = String(season || "");
  return /^\d{8}$/.test(key)
    ? `${key.slice(0, 4)}-${key.slice(4)}`
    : key;
}

function careerCompactSeasonLabel(season) {
  return (
    appState.core.meta.seasons.find((item) => item.key === season)?.label ||
    careerSeasonLabel(season)
  );
}

function careerGroupControl(group) {
  return `
    <div class="field">
      <span class="field-label">Group</span>
      <div class="segmented" data-group-control="career-group">
        <button type="button" data-value="forwards" aria-pressed="${group === "forwards"}">Forwards</button>
        <button type="button" data-value="defense" aria-pressed="${group === "defense"}">Defense</button>
      </div>
    </div>
  `;
}

function careerExplainer(latestSeason) {
  return `
    <details class="methodology-expander career-explainer">
      <summary>
        <span>Understanding the Evolution of a Player's Archetype</span>
        <span class="methodology-toggle" aria-hidden="true"></span>
      </summary>
      <div class="methodology-body">
        <section class="methodology-section">
          <h2>How to Use This Tool</h2>
          <p>
            When you search and select a player, the dropdown shows their
            <strong>name</strong>, <strong>position</strong>, and the
            <strong>most recent season in the dataset</strong> for that player
            (i.e. ${escapeHTML(latestSeason)}).
          </p>
        </section>
        <section class="methodology-section">
          <h2>What Does the Evolution Really Mean?</h2>
          <ul class="career-explainer-list">
            <li><strong>Stable top archetype + high confidence</strong> → consistent role/style across years</li>
            <li><strong>Shifts in top archetype</strong> → role changes, team/system changes, aging, or deployment changes</li>
            <li><strong>Lower confidence</strong> → “mixed profile” seasons where the player blends multiple archetype patterns</li>
          </ul>
        </section>
        <section class="methodology-section">
          <h2>What is Mixedness?</h2>
          <p>In this table, you will see a value for each player called "mixedness". What is that?</p>
          <p>I define <strong>Mixedness</strong> as:</p>
          <div
            class="career-formula"
            role="img"
            aria-label="Mixedness equals one minus the maximum archetype probability for player i"
          >
            <span>Mixedness</span>
            <span>=</span>
            <span>1 − max<sub>k</sub>(p<sub>ik</sub>)</span>
          </div>
          <p>
            where <strong>max<sub>k</sub>(p<sub>ik</sub>)</strong> is the probability
            of the player’s <strong>top archetype</strong> that season.
          </p>
          <ul class="career-explainer-list">
            <li>Mixedness near <strong>0.00</strong> → the model is very confident the player fits a single archetype</li>
            <li>Mixedness &gt;= <strong>0.40</strong> → the player blends multiple archetypes (probability mass is spread out)</li>
          </ul>
        </section>
      </div>
    </details>
  `;
}

function careerHistoryRows(history) {
  const sorted = [...history].sort((a, b) => a.season.localeCompare(b.season));
  return sorted.map((row, index) => {
    const confidencePct =
      row.confidencePct === null || row.confidencePct === undefined
        ? Math.round(Number(row.confidence || 0) * 1000) / 10
        : Number(row.confidencePct);
    const mixedness =
      row.mixedness === null || row.mixedness === undefined
        ? Math.round((1 - Number(row.confidence || 0)) * 1000) / 1000
        : Number(row.mixedness);
    return {
      ...row,
      confidencePct,
      mixedness,
      displaySeason: careerSeasonLabel(row.season),
      compactSeason: careerCompactSeasonLabel(row.season),
      changed:
        index > 0 && row.profile !== sorted[index - 1].profile,
    };
  });
}

function careerConfidenceStatus(value) {
  if (value >= 95) return ["Very stable", "is-very-stable"];
  if (value >= 90) return ["Stable", "is-stable"];
  if (value >= 80) return ["Moderate", "is-moderate"];
  if (value >= 70) return ["Mixed", "is-mixed"];
  return ["Highly mixed", "is-highly-mixed"];
}

function careerMixednessStatus(value) {
  if (value <= 0.05) return ["Very consistent", "is-very-stable"];
  if (value <= 0.1) return ["Consistent", "is-stable"];
  if (value <= 0.2) return ["Some variety", "is-moderate"];
  if (value <= 0.3) return ["Role-shifting", "is-mixed"];
  return ["Blend of styles", "is-highly-mixed"];
}

function careerTeamMarks(teamValue, season) {
  const teams = teamCodes(teamValue);
  if (!teams.length) return "";
  return `
    <span
      class="career-team-marks"
      role="list"
      aria-label="${escapeHTML(`Team logos: ${teams.join(", ")}`)}"
    >
      ${teams
        .map(
          (team) => `
            <span
              class="career-team-mark"
              data-asset-frame
              role="listitem"
              title="${escapeHTML(team)}"
            >
              <span class="career-team-code" aria-hidden="true">${escapeHTML(team)}</span>
              <img
                class="career-team-logo"
                data-asset-sources="${assetSources(teamLogoSources(team, season))}"
                alt=""
                loading="lazy"
                decoding="async"
              />
            </span>
          `,
        )
        .join("")}
    </span>
  `;
}

function careerStat(label, value) {
  return `
    <div>
      <dt>${escapeHTML(label)}</dt>
      <dd>${escapeHTML(value)}</dd>
    </div>
  `;
}

function careerCardStat(label, value) {
  return `
    <div>
      <dt>${escapeHTML(label)}</dt>
      <dd>${escapeHTML(value)}</dd>
    </div>
  `;
}

function careerPlayerStatistics(rows) {
  if (!rows.length) return "";
  const total = (field) =>
    rows.reduce((sum, row) => sum + Number(row[field] || 0), 0);
  const profileCounts = new Map();
  const profileRecency = new Map();
  rows.forEach((row, index) => {
    profileCounts.set(row.profile, (profileCounts.get(row.profile) || 0) + 1);
    profileRecency.set(row.profile, index);
  });
  const mostFrequentProfile =
    [...profileCounts.keys()].sort(
      (left, right) =>
        profileCounts.get(right) - profileCounts.get(left) ||
        profileRecency.get(right) - profileRecency.get(left),
    )[0] || "—";
  const styleChanges = rows.filter((row) => row.changed).length;

  return `
    <section
      class="career-card-summary"
      aria-labelledby="career-card-summary-title"
    >
      <header>
        <div>
          <p class="career-card-eyebrow">Career statistics summary</p>
          <h2 id="career-card-summary-title">Career totals in dataset</h2>
        </div>
      </header>
      <div class="career-card-stat-groups">
        <section class="career-card-stat-group" aria-labelledby="career-card-reg-title">
          <h3 id="career-card-reg-title">Regular season</h3>
          <dl class="career-card-stat-grid">
            ${careerCardStat("GP", number(total("games")))}
            ${careerCardStat("G", number(total("goals")))}
            ${careerCardStat("A", number(total("assists")))}
            ${careerCardStat("P", number(total("points")))}
          </dl>
        </section>
        <section class="career-card-stat-group" aria-labelledby="career-card-po-title">
          <h3 id="career-card-po-title">Playoffs</h3>
          <dl class="career-card-stat-grid">
            ${careerCardStat("GP", number(total("playoffGames")))}
            ${careerCardStat("G", number(total("playoffGoals")))}
            ${careerCardStat("A", number(total("playoffAssists")))}
            ${careerCardStat("P", number(total("playoffPoints")))}
          </dl>
        </section>
      </div>
      <div class="career-card-style">
        <div class="career-card-primary-style">
          <span>Most frequent style</span>
          ${profileChip(mostFrequentProfile)}
        </div>
        <div class="career-card-change-count">
          <span>Style changes</span>
          <strong>${number(styleChanges)}</strong>
        </div>
      </div>
    </section>
  `;
}

function careerSeasonCard(row, index, total) {
  const isLatest = index === total - 1;
  return `
    <li class="career-season-item">
      <details
        class="career-season-card"
        id="career-season-${escapeHTML(row.season)}"
        style="--profile:${profileColor(row.profile)}"
        ${isLatest ? "open" : ""}
      >
        <summary>
          <span class="career-season-node" aria-hidden="true"></span>
          <span class="career-season-summary">
            <span class="career-season-identity">
              <span class="career-season-title-row">
                <span class="career-field-label">Season</span>
                <strong>${escapeHTML(row.displaySeason)}</strong>
                ${row.changed ? '<span class="career-change-label">Profile change</span>' : ""}
              </span>
              <span class="career-team-line">
                <span class="career-field-label">Team(s)</span>
                ${careerTeamMarks(row.team, row.season)}
                <span>${escapeHTML(row.team || "—")}</span>
                <span aria-hidden="true">·</span>
                <span class="career-field-label">Pos</span>
                <span>${escapeHTML(row.position || "UNK")}</span>
              </span>
            </span>
            <span class="career-season-profile">
              <span>Top archetype (season-specific)</span>
              ${profileChip(row.profile)}
            </span>
            <span class="career-season-measures">
              <span>
                <small>Confidence (%)</small>
                <strong>${number(row.confidencePct, 1)}%</strong>
              </span>
              <span>
                <small>Mixedness</small>
                <strong>${number(row.mixedness, 3)}</strong>
              </span>
            </span>
            <span class="career-season-disclosure" aria-hidden="true"></span>
          </span>
        </summary>
        <div class="career-season-detail">
          <section>
            <h3>Regular season</h3>
            <dl class="career-stat-grid">
              ${careerStat("REG GP", number(row.games))}
              ${careerStat("REG ATOI", row.regAtoi || minuteClock(row.toi))}
              ${careerStat("REG P", number(row.points))}
              ${careerStat("REG G", number(row.goals))}
              ${careerStat("REG A", number(row.assists))}
              ${careerStat("REG SOG", number(row.shots))}
              ${careerStat("REG +/-", signedNumber(row.plusMinus))}
              ${careerStat("REG PIM", number(row.pim))}
            </dl>
          </section>
          <section>
            <h3>Playoffs</h3>
            <dl class="career-stat-grid">
              ${careerStat("PO GP", number(row.playoffGames))}
              ${careerStat("PO ATOI", row.playoffAtoi || minuteClock(row.playoffToi))}
              ${careerStat("PO P", number(row.playoffPoints))}
              ${careerStat("PO G", number(row.playoffGoals))}
              ${careerStat("PO A", number(row.playoffAssists))}
              ${careerStat("PO SOG", number(row.playoffShots))}
              ${careerStat("PO +/-", signedNumber(row.playoffPlusMinus))}
              ${careerStat("PO PIM", number(row.playoffPim))}
            </dl>
          </section>
        </div>
      </details>
    </li>
  `;
}

function careerLegend(rows) {
  const profiles = [...new Set(rows.map((row) => row.profile))];
  return `
    <div class="career-profile-legend" aria-label="Top archetype legend">
      <span class="career-legend-title">Top archetype</span>
      ${profiles
        .map(
          (profile) => `
            <span>
              <i style="--profile:${profileColor(profile)}" aria-hidden="true"></i>
              ${escapeHTML(profile)}
            </span>
          `,
        )
        .join("")}
    </div>
  `;
}

function careerTimeline(rows) {
  const chartWidth = Math.max(720, rows.length * 88 + 110);
  return `
    <section class="career-timeline-panel" aria-labelledby="career-timeline-title">
      <header class="career-timeline-head">
        <h2 id="career-timeline-title">Style timeline</h2>
        <ul>
          <li>Hover over data points to see the full details.</li>
          <li>The circled points indicate years where there was a change in player archetype from the previous year.</li>
        </ul>
      </header>
      <div
        class="career-chart-scroll"
        tabindex="0"
        aria-label="Career confidence timeline; scroll horizontally when needed"
      >
        <div
          class="career-chart-stage"
          style="--career-chart-width:${chartWidth}px"
          role="group"
          aria-label="Top-archetype confidence by season"
        >
          <span class="career-y-axis-title" aria-hidden="true">Top-archetype confidence (%)</span>
          <canvas id="career-chart" aria-hidden="true"></canvas>
          <div class="career-chart-points" role="group" aria-label="Career seasons">
            ${rows
              .map(
                (row, index) => `
                  <button
                    class="career-chart-point ${row.changed ? "is-change" : ""}"
                    type="button"
                    data-career-point="${index}"
                    style="--profile:${profileColor(row.profile)}"
                    aria-controls="career-season-${escapeHTML(row.season)}"
                    aria-label="${escapeHTML(
                      `${row.displaySeason}. ${row.profile}. Confidence ${number(row.confidencePct, 1)} percent. Mixedness ${number(row.mixedness, 3)}. Teams ${row.team || "none"}. Position ${row.position || "unknown"}.${row.changed ? " Archetype changed from the previous season." : ""} Open season details.`,
                    )}"
                  ></button>
                `,
              )
              .join("")}
          </div>
          <div class="career-chart-tooltip" role="status" hidden></div>
        </div>
      </div>
      ${careerLegend(rows)}
    </section>
  `;
}

function careerView(rows) {
  if (!rows.length) {
    return '<div class="empty-state">No multi-season data found for this group in data/app/.</div>';
  }
  const avgConfidence = mean(rows.map((row) => row.confidencePct));
  const avgMixedness = mean(rows.map((row) => row.mixedness));
  const [confidenceLabel, confidenceClass] =
    careerConfidenceStatus(avgConfidence);
  const [mixednessLabel, mixednessClass] =
    careerMixednessStatus(avgMixedness);
  return `
    <section class="career-summary-grid" aria-label="Career model summary">
      <article class="career-summary-card">
        <span>Seasons in dataset</span>
        <strong>${number(rows.length)}</strong>
      </article>
      <article class="career-summary-card">
        <span>Avg confidence</span>
        <div>
          <strong>${number(avgConfidence, 1)}%</strong>
          <em class="career-status ${confidenceClass}">${escapeHTML(confidenceLabel)}</em>
        </div>
      </article>
      <article class="career-summary-card">
        <span>Avg mixedness</span>
        <div>
          <strong>${number(avgMixedness, 3)}</strong>
          <em class="career-status ${mixednessClass}">${escapeHTML(mixednessLabel)}</em>
        </div>
      </article>
    </section>
    ${careerTimeline(rows)}
    <section class="career-season-section" aria-labelledby="career-season-title">
      <h2 id="career-season-title">Archetype and Career Stats by Season</h2>
      <ol class="career-season-list">
        ${rows
          .map((row, index) => careerSeasonCard(row, index, rows.length))
          .join("")}
      </ol>
    </section>
  `;
}

function setupCareerTimeline(rows) {
  const stage = document.querySelector(".career-chart-stage");
  const canvas = document.querySelector("#career-chart");
  if (!stage || !canvas || !rows.length) return;
  const context = canvas.getContext("2d");
  const points = [...stage.querySelectorAll("[data-career-point]")];
  const tooltip = stage.querySelector(".career-chart-tooltip");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const listeners = [];
  const listen = (element, type, handler) => {
    element.addEventListener(type, handler);
    listeners.push(() => element.removeEventListener(type, handler));
  };

  function draw() {
    const width = Math.max(stage.clientWidth, 720);
    const height = 320;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const padding = { top: 24, right: 28, bottom: 48, left: 58 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const xAt = (index) =>
      padding.left +
      (rows.length <= 1
        ? plotWidth / 2
        : (plotWidth * index) / (rows.length - 1));
    const yAt = (value) =>
      padding.top + plotHeight - (Math.max(0, Math.min(100, value)) / 100) * plotHeight;

    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const styles = getComputedStyle(document.documentElement);
    const grid = styles.getPropertyValue("--line").trim();
    const muted = styles.getPropertyValue("--muted").trim();
    context.font = "11px Inter, system-ui, sans-serif";
    context.textBaseline = "middle";

    [100, 75, 50, 25, 0].forEach((value) => {
      const y = yAt(value);
      context.beginPath();
      context.setLineDash(value === 0 ? [] : [2, 5]);
      context.strokeStyle = grid;
      context.lineWidth = 1;
      context.moveTo(padding.left, y);
      context.lineTo(width - padding.right, y);
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = muted;
      context.textAlign = "right";
      context.fillText(`${value}%`, padding.left - 10, y);
    });

    rows.forEach((row, index) => {
      context.fillStyle = muted;
      context.textAlign = "center";
      context.fillText(row.compactSeason, xAt(index), height - 17);
    });

    for (let index = 1; index < rows.length; index += 1) {
      const previous = rows[index - 1];
      const current = rows[index];
      const seasonGap =
        Number(current.season.slice(0, 4)) -
          Number(previous.season.slice(0, 4)) >
        1;
      context.beginPath();
      context.setLineDash(seasonGap ? [6, 6] : []);
      context.strokeStyle = "#9ca3af";
      context.lineWidth = 2.5;
      context.lineCap = "round";
      context.moveTo(xAt(index - 1), yAt(previous.confidencePct));
      context.lineTo(xAt(index), yAt(current.confidencePct));
      context.stroke();
      context.setLineDash([]);
    }

    points.forEach((point, index) => {
      point.style.left = `${xAt(index)}px`;
      point.style.top = `${yAt(rows[index].confidencePct)}px`;
    });
    if (tooltip) tooltip.hidden = true;
  }

  function showTooltip(index) {
    if (!tooltip) return;
    const row = rows[index];
    const point = points[index];
    tooltip.innerHTML = `
      <strong>${escapeHTML(row.displaySeason)}</strong>
      <span>${escapeHTML(row.profile)}</span>
      <dl>
        <div><dt>Confidence</dt><dd>${number(row.confidencePct, 1)}%</dd></div>
        <div><dt>Mixedness</dt><dd>${number(row.mixedness, 3)}</dd></div>
        <div><dt>Teams</dt><dd>${escapeHTML(row.team || "—")}</dd></div>
        <div><dt>Pos</dt><dd>${escapeHTML(row.position || "UNK")}</dd></div>
      </dl>
    `;
    tooltip.hidden = false;
    const left = Number.parseFloat(point.style.left || "0");
    const top = Number.parseFloat(point.style.top || "0");
    const tooltipLeft = Math.max(
      8,
      Math.min(
        left - tooltip.offsetWidth / 2,
        stage.clientWidth - tooltip.offsetWidth - 8,
      ),
    );
    const tooltipTop =
      top > tooltip.offsetHeight + 32
        ? top - tooltip.offsetHeight - 18
        : top + 18;
    tooltip.style.left = `${tooltipLeft}px`;
    tooltip.style.top = `${tooltipTop}px`;
  }

  function hideTooltip() {
    if (tooltip) tooltip.hidden = true;
  }

  points.forEach((point, index) => {
    listen(point, "mouseenter", () => showTooltip(index));
    listen(point, "mouseleave", hideTooltip);
    listen(point, "focus", () => showTooltip(index));
    listen(point, "blur", hideTooltip);
    listen(point, "click", () => {
      const seasonCard = document.querySelector(
        `#career-season-${rows[index].season}`,
      );
      if (!seasonCard) return;
      seasonCard.open = true;
      seasonCard.scrollIntoView({
        behavior: reducedMotion ? "auto" : "smooth",
        block: "center",
      });
    });
  });

  draw();
  const observer = new ResizeObserver(draw);
  observer.observe(stage);
  appState.canvasCleanups.push(() => {
    observer.disconnect();
    listeners.forEach((remove) => remove());
  });
}

async function renderCareer() {
  if (!appState.careers) {
    loading("Loading career histories…");
    appState.careers = await getJSON(
      `${DATA_ROOT}/careers.json?v=${DATA_VERSION}`,
    );
  }
  if (appState.route !== "career") return;
  cleanupCanvases();
  const players = careerPlayers(appState.careers, appState.careerGroup);
  if (appState.careerPlayerName) {
    const requestedName = appState.careerPlayerName.trim().toLowerCase();
    const requestedPlayer = players.find(
      (player) => player.name.trim().toLowerCase() === requestedName,
    );
    if (requestedPlayer) appState.careerPlayerId = requestedPlayer.id;
    appState.careerPlayerName = null;
  }
  if (!appState.careerPlayerId || !players.some((player) => player.id === appState.careerPlayerId)) {
    appState.careerPlayerId = players[0]?.id;
  }
  const selected = players.find((player) => player.id === appState.careerPlayerId);
  const history = appState.careers.filter(
    (record) =>
      record.group === appState.careerGroup &&
      record.id === appState.careerPlayerId,
  );
  const rows = careerHistoryRows(history);
  const latest = [...history].sort((a, b) => b.season.localeCompare(a.season))[0];
  const selectedWithTeam = selected
    ? {
        ...selected,
        team: latest?.team || "",
        position: latest?.position || selected.position,
      }
    : null;

  main.innerHTML = `
    <article class="page career-page">
      <header class="career-page-head">
        <h1>How Does a Player's Play Style Evolve Over Their Career?</h1>
        <div class="controls">${careerGroupControl(appState.careerGroup)}</div>
      </header>
      ${careerExplainer(
        careerSeasonLabel(appState.core.meta.seasons[0]?.key),
      )}
      <section class="career-picker" aria-labelledby="career-picker-title">
        <div class="career-picker-controls">
          <h2 id="career-picker-title">Select a player</h2>
          <div class="field">
            <label for="career-search">Search player name</label>
            <input
              class="search-input"
              id="career-search"
              type="search"
              value="${escapeHTML(appState.careerQuery)}"
              autocomplete="off"
              aria-controls="career-matches"
              aria-describedby="career-match-status"
            />
          </div>
          <div class="field">
            <label for="career-matches">Matches</label>
            <select id="career-matches" size="6">
              ${careerSearchResults(players, appState.careerQuery)
                .map(
                  (player) => `
                    <option
                      value="${player.id}"
                      ${player.id === appState.careerPlayerId ? "selected" : ""}
                    >${escapeHTML(player.display)}</option>
                  `,
                )
                .join("")}
            </select>
          </div>
          <p
            class="career-match-status"
            id="career-match-status"
            aria-live="polite"
          ></p>
          <p class="career-no-matches" hidden>No matches. Try a different search.</p>
          <p class="career-selected-caption">
            Selected: ${escapeHTML(selected?.display || "No player")}
          </p>
        </div>
        <div
          class="detail-panel career-selected-player"
          tabindex="-1"
          aria-label="${escapeHTML(`Selected player: ${selected?.name || "No player"}`)}"
        >
          ${
            selectedWithTeam
              ? playerIdentity(
                  selectedWithTeam,
                  latest?.season,
                  "Selected player",
                  `${selectedWithTeam.position} · ${careerSeasonLabel(
                    selected?.firstSeason,
                  )} - ${careerSeasonLabel(selected?.lastSeason)}`,
                )
              : '<div class="detail-name">No player</div>'
          }
          ${selectedWithTeam ? careerPlayerStatistics(rows) : ""}
        </div>
      </section>
      <section id="career-view">${careerView(rows)}</section>
    </article>
  `;
  hydratePlayerAssets(main);

  bindGroupControl("career-group", (group) => {
    appState.careerGroup = group;
    appState.careerPlayerId = null;
    appState.careerPlayerName = null;
    appState.careerQuery = "";
    renderCareer();
  });
  const search = document.querySelector("#career-search");
  const matchesSelect = document.querySelector("#career-matches");
  const matchStatus = document.querySelector("#career-match-status");
  const noMatches = document.querySelector(".career-no-matches");
  const drawMatches = () => {
    appState.careerQuery = search.value;
    const matches = careerSearchResults(players, search.value);
    matchesSelect.innerHTML = matches
      .map(
        (player) => `
          <option
            value="${player.id}"
            ${player.id === appState.careerPlayerId ? "selected" : ""}
          >${escapeHTML(player.display)}</option>
        `,
      )
      .join("");
    matchesSelect.disabled = matches.length === 0;
    matchStatus.textContent = `${number(matches.length)} match${matches.length === 1 ? "" : "es"}`;
    noMatches.hidden = matches.length > 0;
  };
  search.addEventListener("input", drawMatches);
  matchesSelect.addEventListener("change", () => {
    const playerId = Number(matchesSelect.value);
    if (!playerId) return;
    appState.careerPlayerId = playerId;
    appState.careerQuery = "";
    window.history.replaceState(
      null,
      "",
      `#career?player=${encodeURIComponent(playerId)}&group=${encodeURIComponent(appState.careerGroup)}`,
    );
    renderCareer().then(() => {
      document
        .querySelector(".career-selected-player")
        ?.focus({ preventScroll: true });
    });
  });
  drawMatches();
  setupCareerTimeline(rows);
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
  const latest = history.at(-1);
  const selectedWithTeam = selected
    ? {
        ...selected,
        team: latest?.team || "",
        position: latest?.position || "",
      }
    : null;
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
        ${
          selectedWithTeam
            ? playerIdentity(
                selectedWithTeam,
                latest?.season,
                "Selected player",
                `${history.length} playoff season${history.length === 1 ? "" : "s"} in the dataset`,
              )
            : '<div class="detail-name">No player</div>'
        }
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
  hydratePlayerAssets(panel);

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
  const hash = location.hash.replace(/^#\/?/, "") || "overview";
  const [route, queryString = ""] = hash.split("?");
  const routeParams = new URLSearchParams(queryString);
  appState.route = ROUTE_LABELS[route] ? route : "overview";
  if (appState.route === "career") {
    const requestedGroup = routeParams.get("group");
    const requestedPlayer = Number(routeParams.get("player"));
    const requestedName = routeParams.get("name");
    if (requestedGroup === "forwards" || requestedGroup === "defense") {
      appState.careerGroup = requestedGroup;
    }
    if (Number.isInteger(requestedPlayer) && requestedPlayer > 0) {
      appState.careerPlayerId = requestedPlayer;
      appState.careerPlayerName = null;
    } else if (requestedName) {
      appState.careerPlayerName = requestedName;
    }
  }
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
    appState.core = await getJSON(
      `${DATA_ROOT}/core.json?v=${DATA_VERSION}`,
    );
    appState.season = appState.core.meta.seasons[0].key;
    const oldest = appState.core.meta.seasons.at(-1).label;
    const latest = appState.core.meta.seasons[0].label;
    document.querySelector("#coverage-label").textContent =
      `${oldest} to ${latest} · public data`;
    await renderRoute();
  } catch (error) {
    showError(error);
  }
}

window.addEventListener("hashchange", renderRoute);
init();
