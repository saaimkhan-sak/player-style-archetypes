const DATA_ROOT = "/data";
const DATA_VERSION = "20260802-source-closed-v1";
const NEED_GAME_VALUES = [
  0, 5, 10, 15, 20, 25, 30, 35, 40,
  45, 50, 55, 60, 65, 70, 75, 80, 82, 84,
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
  playoffTab: "season",
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
  playoffs: "Playoff Trends",
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

function signedNumber(value, digits = 0) {
  const numeric = Number(value || 0);
  if (numeric > 0) return `+${number(numeric, digits)}`;
  return number(numeric, digits);
}

function mean(values) {
  const valid = values.filter((value) => Number.isFinite(Number(value)));
  return valid.length
    ? valid.reduce((sum, value) => sum + Number(value), 0) / valid.length
    : 0;
}

function median(values) {
  const valid = values
    .map(Number)
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  if (!valid.length) return 0;
  const middle = Math.floor(valid.length / 2);
  return valid.length % 2
    ? valid[middle]
    : (valid[middle - 1] + valid[middle]) / 2;
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
          <p>The v2 style contract uses these canonical signals:</p>
          <ul class="methodology-signals">
            <li><strong>Shot creation and quality:</strong> 5-on-5 shot attempts per 60 and expected goals per attempt</li>
            <li><strong>Interior access:</strong> high-danger shot share and play-continuation rate</li>
            <li><strong>Puck pressure and disruption:</strong> hits, takeaways, giveaways, and blocks per 60</li>
            <li><strong>Two-way context:</strong> on-ice expected goals for and against per 60</li>
            <li><strong>Unknown states:</strong> missing features remain unknown and are not replaced with zero</li>
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
            <span><var>Shots</var>/60</span>
            <span class="equation-symbol">=</span>
            <span class="fraction">
              <span><var>Shots</var></span>
              <span><var>TOI</var><sub>seconds</sub> / 3600</span>
            </span>
          </div>
          <p>Special-teams usage, faceoffs, and scoring outcomes remain available for context tables, but are excluded from the v2 style fingerprint.</p>
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
            <var>W</var><var>H</var>
          </div>
          <p>
            You can think of each row of <var>W</var> as a compact
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
            <span>∑<sub><var>k</var>=1</sub><sup><var>K</var></sup> <var>π</var><sub><var>k</var></sub> N(<var>z</var> | <var>μ</var><sub><var>k</var></sub>, <var>Σ</var><sub><var>k</var></sub>)</span>
          </div>
          <p>For each player (<var>i</var>), the model outputs a membership weight for each learned style:</p>
          <div
            class="equation equation-model"
            role="math"
            aria-label="p i k equals the membership weight for style k given z i"
          >
            <span><var>p</var><sub><var>i</var><var>k</var></sub></span>
            <span class="equation-symbol">=</span>
            <span>P(Archetype = <var>k</var> | <var>z</var><sub><var>i</var></sub>)</span>
          </div>
        </section>

        <section
          class="methodology-section methodology-interpretation"
          aria-labelledby="method-blends"
        >
          <h2 id="method-blends">Why do blended style profiles exist?</h2>
          <p>
            The model is a soft clustering system: instead of forcing every player into
            exactly one bucket, it assigns a membership weight across learned styles.
            Some players genuinely combine traits that sit between multiple clusters
            (e.g., moderate scoring + moderate physical play), so their profile names
            describe the strongest trait combination rather than pretending every
            cluster is one clean role.
          </p>
          <p>
            <strong>Interpretation:</strong> if a player’s membership weights are
            (0.1, 87.3, 6.4, 6.3)%, the largest share is aligned to one learned style.
            This is a within-season model fit signal, not a probability that the
            hockey interpretation is correct.
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
  const context = canvas.getContext("2d");
  const wrapper = canvas.parentElement;
  const referenceImage = new Image();
  const referenceWidth = 240;
  const referenceHeight = 268;

  function drawReferenceRink() {
    const width = Math.max(wrapper.clientWidth, 250);
    const height = Math.max(wrapper.clientHeight, 150);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const scale = Math.min((width - 14) / referenceWidth, (height - 14) / referenceHeight);
    const offsetX = (width - referenceWidth * scale) / 2;
    const offsetY = (height - referenceHeight * scale) / 2;
    context.setTransform(
      ratio * scale,
      0,
      0,
      ratio * scale,
      ratio * offsetX,
      ratio * offsetY,
    );
    if (referenceImage.complete && referenceImage.naturalWidth) {
      context.drawImage(referenceImage, 0, 0, referenceWidth, referenceHeight);
    }
  }

  referenceImage.addEventListener("load", drawReferenceRink);
  referenceImage.src = "data/hero-rink-reference.png";
  drawReferenceRink();
  const referenceObserver = new ResizeObserver(drawReferenceRink);
  referenceObserver.observe(wrapper);
  appState.canvasCleanups.push(() => referenceObserver.disconnect());
  return;

  const styles = getComputedStyle(document.documentElement);
  const colors = {
    board: "#2c3b53",
    ice: "#fffefa",
    red: "#c92525",
    redLine: "#bc8d9a",
    blue: "#9caab4",
    ink: "#111820",
    logo: "#8997a0",
  };
  const artWidth = 240;
  const artHeight = 268;

  function draw() {
    const width = Math.max(wrapper.clientWidth, 250);
    const height = Math.max(wrapper.clientHeight, 150);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const scale = Math.min((width - 14) / artWidth, (height - 14) / artHeight);
    const offsetX = (width - artWidth * scale) / 2;
    const offsetY = (height - artHeight * scale) / 2;
    context.setTransform(
      ratio * scale,
      0,
      0,
      ratio * scale,
      ratio * offsetX,
      ratio * offsetY,
    );

    const rink = { left: 2, top: 16, width: 238, height: 236, radius: 58 };
    function leftHalfPath(target) {
      target.beginPath();
      target.moveTo(58, rink.top);
      target.lineTo(240, rink.top);
      target.lineTo(240, rink.top + rink.height);
      target.lineTo(58, rink.top + rink.height);
      target.quadraticCurveTo(rink.left, rink.top + rink.height, rink.left, rink.top + rink.height - rink.radius);
      target.lineTo(rink.left, rink.top + rink.radius);
      target.quadraticCurveTo(rink.left, rink.top, 58, rink.top);
      target.closePath();
    }

    function boardOutlinePath(target) {
      target.beginPath();
      target.moveTo(58, rink.top);
      target.lineTo(240, rink.top);
      target.moveTo(240, rink.top + rink.height);
      target.lineTo(58, rink.top + rink.height);
      target.quadraticCurveTo(rink.left, rink.top + rink.height, rink.left, rink.top + rink.height - rink.radius);
      target.lineTo(rink.left, rink.top + rink.radius);
      target.quadraticCurveTo(rink.left, rink.top, 58, rink.top);
    }

    leftHalfPath(context);
    context.save();
    context.clip();
    context.fillStyle = colors.ice;
    context.fillRect(0, 0, artWidth, artHeight);

    function verticalLine(position, color, thickness = 2) {
      context.fillStyle = color;
      context.fillRect(position - thickness / 2, rink.top, thickness, rink.height);
    }

    verticalLine(171, colors.blue, 3);
    verticalLine(240, colors.redLine, 3);

    function faceoffDot(cx, cy, radius = 2) {
      context.fillStyle = colors.redLine;
      context.beginPath();
      context.arc(cx, cy, radius, 0, Math.PI * 2);
      context.fill();
    }

    function faceoffMarking(cx, cy) {
      faceoffDot(cx, cy);
      context.strokeStyle = colors.redLine;
      context.lineWidth = 1.5;
      const inner = 5.5;
      const outer = 12;
      const top = 8;
      const upperBar = 1;
      const lowerBar = 3;
      const bottom = 10;
      [
        [[cx - outer, cy - upperBar], [cx - inner, cy - upperBar], [cx - inner, cy - top]],
        [[cx + inner, cy - top], [cx + inner, cy - upperBar], [cx + outer, cy - upperBar]],
        [[cx - outer, cy + lowerBar], [cx - inner, cy + lowerBar], [cx - inner, cy + bottom]],
        [[cx + inner, cy + lowerBar], [cx + outer, cy + lowerBar], [cx + inner, cy + bottom]],
      ].forEach((segments) => {
        context.beginPath();
        context.moveTo(segments[0][0], segments[0][1]);
        context.lineTo(segments[1][0], segments[1][1]);
        context.lineTo(segments[2][0], segments[2][1]);
        context.stroke();
      });
    }

    function faceoffCircle(cx, cy) {
      context.strokeStyle = colors.redLine;
      context.lineWidth = 1.7;
      context.beginPath();
      context.arc(cx, cy, 38, 0, Math.PI * 2);
      context.stroke();
      faceoffMarking(cx, cy);

      // The reference uses two short vertical hashmarks above and below each circle.
      context.lineWidth = 1.5;
      [[cx - 7.5, cy - 44, cx - 7.5, cy - 37],
        [cx + 7.5, cy - 44, cx + 7.5, cy - 37],
        [cx - 7.5, cy + 37, cx - 7.5, cy + 44],
        [cx + 7.5, cy + 37, cx + 7.5, cy + 44]].forEach(([x1, y1, x2, y2]) => {
        context.beginPath();
        context.moveTo(x1, y1);
        context.lineTo(x2, y2);
        context.stroke();
      });
    }

    [76, 193].forEach((cy) => faceoffCircle(84, cy));
    [76, 193].forEach((cy) => faceoffDot(185, cy, 2.2));

    // The center circle is centered on the red line, so only its left half is visible.
    context.strokeStyle = colors.redLine;
    context.lineWidth = 1.7;
    context.beginPath();
    context.arc(240, 134, 38, 0, Math.PI * 2);
    context.stroke();
    context.fillStyle = colors.logo;
    context.font = "700 italic 16px Arial, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText("IHS", 240, 135);

    function drawNet(xPosition) {
      context.strokeStyle = colors.red;
      context.lineWidth = 1.5;
      context.strokeRect(xPosition, 123, 12, 23);
      context.beginPath();
      context.moveTo(xPosition + 4, 123);
      context.lineTo(xPosition + 4, 146);
      context.moveTo(xPosition + 8, 123);
      context.lineTo(xPosition + 8, 146);
      context.stroke();
    }

    drawNet(25);

    function arrowHead(xPosition, yPosition, angle, color, size = 8) {
      context.save();
      context.fillStyle = color;
      context.beginPath();
      context.moveTo(xPosition, yPosition);
      context.lineTo(
        xPosition - Math.cos(angle - Math.PI / 6) * size,
        yPosition - Math.sin(angle - Math.PI / 6) * size,
      );
      context.lineTo(
        xPosition - Math.cos(angle + Math.PI / 6) * size,
        yPosition - Math.sin(angle + Math.PI / 6) * size,
      );
      context.closePath();
      context.fill();
      context.restore();
    }

    function curveArrow(start, controlOne, controlTwo, end, color, dashed = false) {
      context.save();
      context.strokeStyle = color;
      context.lineWidth = 1.6;
      context.lineCap = "round";
      context.setLineDash(dashed ? [4, 4] : []);
      context.beginPath();
      context.moveTo(start[0], start[1]);
      context.bezierCurveTo(
        controlOne[0], controlOne[1], controlTwo[0], controlTwo[1], end[0], end[1],
      );
      context.stroke();
      context.setLineDash([]);
      const angle = Math.atan2(end[1] - controlTwo[1], end[0] - controlTwo[0]);
      arrowHead(end[0], end[1], angle, color, 7);
      context.restore();
    }

    function straightArrow(start, end, color, dashed = false) {
      context.save();
      context.strokeStyle = color;
      context.lineWidth = 1.5;
      context.setLineDash(dashed ? [4, 4] : []);
      context.beginPath();
      context.moveTo(start[0], start[1]);
      context.lineTo(end[0], end[1]);
      context.stroke();
      context.setLineDash([]);
      arrowHead(end[0], end[1], Math.atan2(end[1] - start[1], end[0] - start[0]), color, 7);
      context.restore();
    }

    function marker(label, cx, cy, fill, textColor = "#fff", radius = 8) {
      context.fillStyle = fill;
      context.beginPath();
      context.arc(cx, cy, radius, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = textColor;
      context.font = `700 ${label.length > 1 ? 8 : 9}px Arial, sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(label, cx, cy + 0.5);
    }

    // The reference play: black X/O markers, red forwards/defenders, and their routes.
    straightArrow([111, 31], [161, 69], colors.red, true);
    straightArrow([168, 78], [168, 128], colors.red);
    straightArrow([165, 239], [111, 244], colors.red);
    straightArrow([64, 219], [88, 243], colors.ink, true);
    curveArrow([49, 91], [31, 86], [35, 102], [77, 126], colors.red);
    curveArrow([103, 99], [120, 110], [103, 112], [112, 128], colors.red);
    curveArrow([126, 177], [107, 179], [113, 151], [118, 144], colors.red);
    curveArrow([119, 186], [119, 216], [146, 235], [163, 242], colors.red);

    marker("X", 90, 32, colors.ink);
    marker("X", 22, 93, colors.ink);
    marker("G", 43, 134, colors.ink);
    marker("X", 94, 140, colors.ink);
    marker("X", 20, 209, colors.ink);
    marker("X", 88, 246, colors.ink);
    marker("F2", 92, 93, colors.red);
    marker("F3", 146, 134, colors.red);
    marker("F1", 91, 174, colors.red);
    marker("D2", 167, 73, colors.red);
    marker("D1", 164, 244, colors.red);

    context.restore();
    boardOutlinePath(context);
    context.strokeStyle = colors.board;
    context.lineWidth = 2.4;
    context.stroke();
    context.strokeStyle = colors.redLine;
    context.lineWidth = 2.2;
    context.beginPath();
    context.moveTo(240, rink.top);
    context.lineTo(240, rink.top + rink.height);
    context.stroke();
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

      <section class="metric-grid overview-metric-grid" aria-label="Dataset summary">
        <div class="metric metric-coverage">
          <span class="metric-label">Season coverage</span>
          <span class="metric-value metric-season-range">
            <span>${escapeHTML(oldest)}</span>
            <span><small>to</small> ${escapeHTML(latest)}</span>
          </span>
        </div>
        ${metric("Players analyzed", number(meta.playerCount), "NHL players")}
        ${metric("Different styles", number(meta.namedStyleCount), `${meta.namedStyleBreakdown.forwards} forward · ${meta.namedStyleBreakdown.defense} defense; season-derived`)}
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
          <div class="definition-callout">
            <h3>Freshness and comparability</h3>
            <p><strong>Latest complete style season:</strong> ${escapeHTML(meta.latestCompleteSeason || "20252026")}</p>
            <p><strong>Regular-season evidence through:</strong> ${escapeHTML(meta.regularSeasonDataThrough || "—")}</p>
            <p><strong>Playoff evidence through:</strong> ${escapeHTML(meta.playoffDataThrough || "—")}</p>
            <p><strong>Upcoming NHL season:</strong> 2026–27, 84 games per club. Season-derived labels are not treated as stable career identities.</p>
          </div>
          <div class="definition-callout release-evidence-callout">
            <h3>Release evidence</h3>
            <p>The current 2025–26 release reconciles 1,394 official NHL regular/playoff game IDs: 1,312 regular-season games and 82 playoff games through the June 14, 2026 Stanley Cup Final.</p>
            <p>Playoff projections use direct MoneyPuck game-by-game player files plus official NHL game records. <a href="/data/manifest.json" target="_blank" rel="noreferrer">Open the release manifest</a> or <a href="/data/freshness.json" target="_blank" rel="noreferrer">freshness record</a>.</p>
          </div>
        </section>

        <section class="story-section methods-section" id="methods">
          <div class="story-heading">
            <h2>Methods</h2>
          </div>
          <p class="section-intro">At a high level, I’m learning a “style fingerprint” for each player-season using public data, then clustering those fingerprints into archetypes.</p>

          <div class="data-used">
            <h3>Data used</h3>
            <p>From official NHL game endpoints I aggregate per player:</p>
            <ul>
              <li>regular season vs playoff statistics</li>
              <li>time on ice, game state, teams, scores, and game dates</li>
              <li>canonical boxscore counting stats (hits/blocks/takeaways/giveaways plus context outcomes)</li>
            </ul>
            <p>From MoneyPuck’s listed player game-by-game downloads I add advanced regular- and playoff-season signals. MoneyPuck data is used for this noncommercial project with clear credit; saved page HTML is not a production input.</p>
            <ul>
              <li>5-on-5 shot-attempt rate and expected goals per attempt</li>
              <li>high-danger shot share and play-continuation rate</li>
              <li>on-ice expected goals for/against rates</li>
              <li>game-level coverage, retrieval date, hashes, and explicit unknown states</li>
            </ul>
            <p>Special-teams usage, faceoffs, and scoring outcomes remain available for context tables, but are excluded from the v2 style fingerprint so role and outcome fields do not duplicate the style signal.</p>
            <p>Because those MoneyPuck files start in 2008, the site focuses on seasons from 2008-09 forward.</p>
          </div>

          <div class="method-stack">
            <article class="method-detail">
              <span>01</span>
              <div>
                <h3>Step 1 — Normalize for ice time (so players are comparable)</h3>
                <p>Players have different ice time, so I convert raw counts into per-60 rates:</p>
                <div class="formula" aria-label="Shots per 60 equals shots divided by time on ice in seconds divided by 3600">
                  <span><var>Shots</var>/60</span>
                  <b>=</b>
                  <span><var>Shots</var> ÷ (<var>TOI</var><sub>seconds</sub> / 3600)</span>
                </div>
                <p>Special-teams usage, faceoffs, and scoring outcomes are retained for context and audit tables, but are not part of the v2 style-learning feature blocks.</p>
              </div>
            </article>

            <article class="method-detail">
              <span>02</span>
              <div>
                <h3>Step 2 — Put all features on the same scale</h3>
                <p>Some stats have heavy tails. To keep a few extreme values from dominating, I use a robust scaling transformation:</p>
                <div class="formula" aria-label="x star equals x minus median of x divided by the interquartile range of x">
                  <span><var>x</var><sup>*</sup></span>
                  <b>=</b>
                  <span>(<var>x</var> − median(<var>x</var>)) ÷ IQR(<var>x</var>)</span>
                </div>
              </div>
            </article>

            <article class="method-detail">
              <span>03</span>
              <div>
                <h3>Step 3 — Compress into a smaller “style fingerprint”</h3>
                <p>To summarize correlated features, I use Non-negative Matrix Factorization (NMF):</p>
                <div class="formula" aria-label="X approximately equals W H">
                  <span><var>X</var></span>
                  <b>≈</b>
                  <span><var>W</var><var>H</var></span>
                </div>
                <p>Think of each row of <var>W</var> as a compact “style fingerprint” describing how a player produces their results.</p>
              </div>
            </article>

            <article class="method-detail">
              <span>04</span>
              <div>
                <h3>Step 4 — Learn archetypes with a probabilistic clustering model</h3>
                <p>I fit a Gaussian Mixture Model (GMM) to the fingerprints:</p>
                <div class="formula formula-wide" aria-label="p of z equals the sum from k equals 1 to K of pi k times a normal distribution">
                  <span><var>p</var>(<var>z</var>)</span>
                  <b>=</b>
                  <span>Σ<sub><var>k</var>=1</sub><sup><var>K</var></sup> <var>π</var><sub><var>k</var></sub> N(<var>z</var> | <var>μ</var><sub><var>k</var></sub>, <var>Σ</var><sub><var>k</var></sub>)</span>
                </div>
                <p>For each player-season, the model outputs style-membership weights using this formula. They describe proximity to learned clusters, not the probability that a hockey interpretation is correct.</p>
                <div class="formula formula-wide">
                  <span><var>p</var><sub><var>i</var><var>k</var></sub></span>
                  <b>=</b>
                  <span>P(Archetype = <var>k</var> | <var>z</var><sub><var>i</var></sub>)</span>
                </div>
                <p>Because this is soft clustering, a player can be “70% Playmaking Scorer, 20% Two-Way Creator, 10% Role Specialist” rather than being forced into a single bucket.</p>
                <p>I summarize the share outside the top style as a blend indicator:</p>
                <div class="formula formula-wide">
                  <span>Style blend</span>
                  <b>=</b>
                  <span>1 − max<sub><var>k</var></sub>(<var>p</var><sub><var>i</var><var>k</var></sub>)</span>
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
        <p class="data-disclaimer">This page is a dated release. Check the freshness record and source manifest before treating a season as complete.</p>
      </div>
    </article>
  `;

  setupHeroRink(document.querySelector("#hero-rink"));
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
          label: "Learned styles",
          value: number(groupData.profiles.length),
        },
        {
          label: "Player-seasons",
          value: number(groupData.players.length),
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
      ${metric("Learned styles", number(groupData.profiles.length), "season-derived style categories")}
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
        <div class="detail-stat"><span>Top-style share</span><strong>${percent(player.confidence)}</strong></div>
      </div>
      <p class="detail-caption">Membership weights across learned styles; this is a within-season fit signal, not a correctness probability.</p>
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
            <th class="numeric">Top-style share</th>
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
    <span class="result-count">Select a name for the full style-membership mix.</span>
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
                      ${number(Number(player.confidence || 0) * 100, 1)}% top-style share
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
            <th class="numeric">Top-style share</th>
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
  const maxGames = Number(
    appState.core?.meta?.scheduleDimension?.[appState.season] || 82,
  );
  const values = NEED_GAME_VALUES.filter((value) => value <= maxGames);
  if (!values.includes(maxGames)) values.push(maxGames);
  const index = Math.max(
    0,
    Math.min(values.length - 1, Number(control?.value || 0)),
  );
  return values[index];
}

function needGameOptions() {
  const maxGames = Number(
    appState.core?.meta?.scheduleDimension?.[appState.season] || 82,
  );
  const values = NEED_GAME_VALUES.filter((value) => value <= maxGames);
  if (!values.includes(maxGames)) values.push(maxGames);
  return { maxGames, values };
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
          <span class="need-summary-label">Profile fit</span>
          <strong>${number(targetSimilarity, 1)}%</strong>
          <span
            class="need-similarity-track"
            role="progressbar"
            aria-label="${escapeHTML(player.name)} profile fit"
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
            ${number(Number(player.confidence || 0) * 100, 1)}% top-style share
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
  const { maxGames, values } = needGameOptions();
  const defaultGames = Math.min(20, maxGames);
  const defaultIndex = Math.max(0, values.indexOf(defaultGames));
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
            <li>A ranked list of players who best match a selected statistical style profile.</li>
          </ul>
        </div>
        <div>
          <h3>How to use it</h3>
          <ul>
            <li>Pick the archetype you want to add to a roster.</li>
            <li>Optionally exclude your own team.</li>
            <li>Increase minimum regular-season games to avoid tiny samples.</li>
            <li>“Profile fit (%)” is a within-season fitted match score, not a probability that the hockey interpretation is correct.</li>
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
            <label for="need-games">Min REG games (max ${maxGames})</label>
            <output id="need-games-value" for="need-games">${defaultGames}</output>
        </div>
        <input
          id="need-games"
          type="range"
          min="0"
          max="${values.length - 1}"
          value="${defaultIndex}"
          step="1"
          aria-valuemin="0"
          aria-valuemax="${maxGames}"
          aria-valuenow="${defaultGames}"
          aria-valuetext="${defaultGames} regular-season games"
        />
        <div class="need-range-ends" aria-hidden="true"><span>0</span><span>${maxGames}</span></div>
      </div>
    </section>
    <div id="need-results" aria-live="polite">${needResults(groupData, target.cluster, target.profile, "", defaultGames)}</div>
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
          <p>Ordered by fitted profile fit, then regular-season points. Model version and data reliability should be checked before using a match.</p>
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
            <li><strong>Stable top archetype + high membership concentration</strong> → consistent role/style across years</li>
            <li><strong>Shifts in top archetype</strong> → role changes, team/system changes, aging, or deployment changes</li>
            <li><strong>Lower membership concentration</strong> → blended seasons where the player shares traits with multiple learned styles</li>
          </ul>
        </section>
        <section class="methodology-section">
          <h2>What is Style Blend?</h2>
          <p>In this table, the style-blend value shows how much membership sits outside the player's top learned style.</p>
          <p>I define <strong>Style blend</strong> as:</p>
          <div
            class="career-formula"
            role="img"
            aria-label="Style blend equals one minus the maximum archetype membership for player i"
          >
            <span>Style blend</span>
            <span>=</span>
            <span>1 − max<sub>k</sub>(p<sub>ik</sub>)</span>
          </div>
          <p>
            where <strong>max<sub>k</sub>(p<sub>ik</sub>)</strong> is the membership
            share of the player’s <strong>top archetype</strong> that season.
          </p>
          <ul class="career-explainer-list">
            <li>Style blend near <strong>0.00</strong> → most membership sits in one learned style</li>
            <li>Style blend &gt;= <strong>0.40</strong> → membership is spread across multiple learned styles</li>
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
        <p class="career-card-eyebrow" id="career-card-summary-title">
          Career totals in dataset
        </p>
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
                <small>Top-style share (%)</small>
                <strong>${number(row.confidencePct, 1)}%</strong>
              </span>
              <span>
                <small>Style blend</small>
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
        aria-label="Career top-style share timeline; scroll horizontally when needed"
      >
        <div
          class="career-chart-stage"
          style="--career-chart-width:${chartWidth}px"
          role="group"
          aria-label="Top-archetype share by season"
        >
          <span class="career-y-axis-title" aria-hidden="true">Top-archetype share (%)</span>
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
                      `${row.displaySeason}. ${row.profile}. Top-style share ${number(row.confidencePct, 1)} percent. Style blend ${number(row.mixedness, 3)}. Teams ${row.team || "none"}. Position ${row.position || "unknown"}.${row.changed ? " Archetype changed from the previous season." : ""} Open season details.`,
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
        <span>Avg top-style share</span>
        <div>
          <strong>${number(avgConfidence, 1)}%</strong>
          <em class="career-status ${confidenceClass}">${escapeHTML(confidenceLabel)}</em>
        </div>
      </article>
      <article class="career-summary-card">
        <span>Avg style blend</span>
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
        <div><dt>Top-style share</dt><dd>${number(row.confidencePct, 1)}%</dd></div>
        <div><dt>Style blend</dt><dd>${number(row.mixedness, 3)}</dd></div>
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
                  `${selectedWithTeam.position}
${careerSeasonLabel(
                    selected?.firstSeason,
                  )} to ${careerSeasonLabel(selected?.lastSeason)}`,
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

function playoffBaseRows() {
  return appState.playoffs.filter(
    (row) =>
      row.group === appState.playoffGroup &&
      row.regGames >= appState.playoffMinReg &&
      row.playoffGames >= appState.playoffMinPo,
  );
}

function playoffSeasonRows(base) {
  return base.filter((row) => row.season === appState.playoffSeason);
}

function playoffSeasonLabel(season) {
  return careerSeasonLabel(season);
}

function playoffControls() {
  const available = [
    ...new Set(
      appState.playoffs
        .filter((row) => row.playoffGames > 0)
        .map((row) => row.season),
    ),
  ]
    .sort()
    .reverse();
  return `
    <div class="field">
      <label for="playoff-season">Season</label>
      <select id="playoff-season">
        ${available
          .map(
            (season) => `
              <option
                value="${season}"
                ${season === appState.playoffSeason ? "selected" : ""}
              >${escapeHTML(playoffSeasonLabel(season))}</option>
            `,
          )
          .join("")}
      </select>
    </div>
    <div class="field">
      <span class="field-label">Group</span>
      <div class="segmented" data-group-control="playoff-group">
        <button type="button" data-value="forwards" aria-pressed="${appState.playoffGroup === "forwards"}">Forwards</button>
        <button type="button" data-value="defense" aria-pressed="${appState.playoffGroup === "defense"}">Defense</button>
      </div>
    </div>
  `;
}

function playoffExplainer() {
  return `
    <details class="methodology-expander playoff-methodology">
      <summary>
        <span>📊 How is the model shift score calculated? (click to expand)</span>
        <span class="methodology-toggle" aria-hidden="true"></span>
      </summary>
      <div class="methodology-body">
        <section class="methodology-section">
          <h2>The short version</h2>
          <p>
            I took each player's playoff statistics, ran them through the exact
            same machine-learning model used to assign regular-season archetypes,
            and measured how far the player's playoff "style fingerprint" is from
            their regular-season one. A bigger number = a bigger identity shift.
          </p>
        </section>
        <section class="methodology-section">
          <h2>Step 1 — Where the data comes from</h2>
          <p>
            <strong>Regular season:</strong> The archetype model was trained on
            <a href="https://moneypuck.com" target="_blank" rel="noreferrer">MoneyPuck</a>
            player-level advanced metrics — a well-regarded public data source that
            tracks things like expected goals (xGoals), shot quality, and on-ice
            possession at the individual player level, game by game.
          </p>
          <p>
            <strong>Playoffs:</strong> the current release uses MoneyPuck's listed
            player game-by-game download together with official NHL game records.
            The 2025–26 source register reconciles every regular-season and playoff
            game ID through the June 14, 2026 Stanley Cup Final. The model uses the
            same canonical style features in both contexts:
          </p>
          <ul>
            <li><strong>5-on-5 shot creation and shot quality</strong></li>
            <li><strong>high-danger access and play continuation</strong></li>
            <li><strong>canonical NHL disruption rates</strong> — hits, takeaways, giveaways, and blocks</li>
            <li><strong>on-ice expected goals for and against</strong></li>
          </ul>
        </section>
        <section class="methodology-section">
          <h2>Step 2 — What I calculated from the playoff data</h2>
          <p>
            For each player and each situation, I computed the same types of rate
            statistics the regular-season model uses:
          </p>
          <div class="playoff-method-table-wrap">
            <table class="playoff-method-table">
              <thead>
                <tr><th>Metric</th><th>What it measures</th><th>Why it matters</th></tr>
              </thead>
              <tbody>
                <tr><td><strong>Expected goals per shot attempt (5v5)</strong></td><td>The average expected-goal value of a player's attempts</td><td>Captures shot-quality style without treating goals as a style input</td></tr>
                <tr><td><strong>Shot attempts per 60 min (5v5)</strong></td><td>How frequently a player gets involved in shooting plays</td><td>Captures offensive pressure regardless of whether shots go in</td></tr>
                <tr><td><strong>High-danger shot share</strong></td><td>What fraction of a player's shots come from the most dangerous areas (in tight, directly in front)</td><td>Identifies net-front finishers vs perimeter shooters</td></tr>
                <tr><td><strong>Play continuation per 60 (5v5)</strong></td><td>How often a player's offensive-zone sequence continues</td><td>Captures possession continuation and puck support</td></tr>
                <tr><td><strong>On-ice xGoals For/Against per 60 (5v5)</strong></td><td>How good/bad the team was at creating and allowing expected goals <em>while this player was on the ice</em></td><td>Measures two-way impact and deployment quality</td></tr>
                <tr><td><strong>Hits, takeaways, giveaways, and blocks per 60</strong></td><td>Canonical NHL physical and puck-battle contributions</td><td>Separates disruption styles without duplicating MoneyPuck event fields</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            Playoff rows are built from the permitted MoneyPuck game-by-game
            download and official NHL game records. Missing playoff features are
            treated as unknown and do not copy regular-season values. Rows require
            at least five playoff games and 150 eligible seconds before publication.
          </p>
        </section>
        <section class="methodology-section">
          <h2>Step 3 — Running it through the model</h2>
          <p>
            <strong>NMF compression:</strong> Non-negative Matrix Factorization
            squashes all those metrics into a compact "style fingerprint" — a short
            list of numbers that describe <em>how</em> a player plays rather than
            <em>how much</em> they produce. Think of it as distilling a player's
            full stat line into a few key style dimensions.
          </p>
          <p>
            <strong>GMM classification:</strong> A Gaussian Mixture Model then takes
            that fingerprint and outputs a <em>membership distribution</em> across
            learned styles. For example, a player might be represented as:
          </p>
          <ul>
            <li>Regular season: 72% membership in Playmaking Scorer, 18% in Two-Way Creator, 10% elsewhere</li>
            <li>Playoffs: 41% membership in Playmaking Scorer, 44% in Two-Way Creator, 15% elsewhere</li>
          </ul>
          <p>
            The regular-season model was not re-trained on playoff data — I used
            the same fitted model to project each player into archetype space based
            on their playoff numbers.
          </p>
        </section>
        <section class="methodology-section">
          <h2>Step 4 — The model shift score</h2>
          <p>
            The <strong>model shift score</strong> is the Euclidean distance between
            those two membership distributions:
          </p>
          <blockquote>
            <em>How much did the player's style-membership profile move from regular season to playoffs?</em>
          </blockquote>
          <ul>
            <li><strong>Score near 0</strong> → The model sees essentially the same player in both contexts. The style fingerprint barely changed.</li>
            <li><strong>Score around 0.25–0.75</strong> → Moderate shift. The player looks meaningfully different — perhaps leaning into a different role or responding to matchup adjustments.</li>
            <li><strong>Score above 0.75</strong> → Major shift. The playoff version of this player would likely be classified into a different archetype than the regular-season version.</li>
          </ul>
        </section>
        <section class="methodology-section">
          <h2>What about the "stat shift score"?</h2>
          <p>
            The table also shows a simpler <strong>stat shift score</strong> that was
            used before the advanced data was available. It compares raw boxscore
            metrics (points per game, shots per game, ice time, penalty minutes,
            plus/minus) between regular season and playoffs using z-scores, then
            combines them. It is less informative than the model shift score because:
          </p>
          <ol>
            <li>Scoring rates <em>universally</em> decline in the playoffs due to tighter play and better goaltending — so a drop in P/GP doesn't necessarily mean a player changed their style</li>
            <li>It doesn't capture shot quality, on-ice possession, or zone-start context</li>
          </ol>
          <p>
            The model shift score is a descriptive comparison, not proof that a
            player’s underlying style changed. Treat it as a sample-dependent
            profile movement signal and check the reliability grade beside it.
          </p>
        </section>
        <section class="methodology-section">
          <h2>Limitations</h2>
          <ul>
            <li>Playoff sample sizes are smaller than regular-season totals, adding noise — especially for players eliminated in round one</li>
            <li>The model was trained on regular-season distributions, which are slightly wider than playoff distributions (extreme performers are more common in the regular season). This means the model is working slightly "out of sample" when applied to playoffs</li>
            <li>Playoff evidence is sample-limited; missing features remain unknown rather than being copied from the regular season.</li>
          </ul>
        </section>
      </div>
    </details>
  `;
}

function playoffPrimarySignal() {
  return `
    <aside class="playoff-primary-signal">
      <strong>Primary signal: Model shift score</strong>
      <p>
        — measures how far a player's playoff style fingerprint moves in archetype
        space, based on 5-on-5 shot creation, shot quality, continuation,
        disruption, and on-ice expected-goal rates.
        <strong>Higher = bigger identity shift.</strong> The scatter plot shape
        encodes shift band; point size encodes playoff games played.
      </p>
    </aside>
  `;
}

function playoffBandPill(band) {
  const label = band || "Not projected";
  const className = {
    "Held steady": "is-steady",
    "Moderate shift": "is-moderate",
    "Major shift": "is-major",
  }[label] || "is-unavailable";
  return `<span class="playoff-band ${className}">${escapeHTML(label)}</span>`;
}

function playoffScorePill(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '<span class="playoff-score is-unavailable">—</span>';
  }
  const className =
    numeric >= 0.75
      ? "is-major"
      : numeric >= 0.25
        ? "is-moderate"
        : "is-steady";
  return `
    <span class="playoff-score ${className}">
      ${number(numeric, 3)}
    </span>
  `;
}

function playoffShiftTable(rows, limit = 30, order = "shift") {
  const sorted = [...rows]
    .sort((left, right) =>
      order === "season"
        ? left.season.localeCompare(right.season)
        : Number(right.distance) - Number(left.distance),
    )
    .slice(0, limit);
  return `
    <div class="playoff-table-wrap" tabindex="0" aria-label="Playoff profile changes table; scroll horizontally for all columns">
      <table class="playoff-table">
        <thead>
          <tr>
            <th>Season</th>
            <th>Player</th>
            <th>Team(s)</th>
            <th>Pos</th>
            <th class="playoff-archetype-column">REG archetype</th>
            <th class="playoff-archetype-column">Projected PO archetype</th>
            <th class="numeric">REG GP</th>
            <th class="numeric">PO GP</th>
            <th class="numeric">REG P/GP</th>
            <th class="numeric">PO P/GP</th>
            <th class="numeric">P/GP change</th>
            <th class="numeric">TOI change</th>
            <th class="numeric">Model shift ↑</th>
            <th>Model shift band</th>
            <th>Sample reliability</th>
            <th class="numeric">Stat shift</th>
            <th>Stat shift band</th>
            <th>REG ATOI</th>
            <th>PO ATOI</th>
          </tr>
        </thead>
        <tbody>
          ${sorted
            .map(
              (row) => `
                <tr>
                  <td>${escapeHTML(playoffSeasonLabel(row.season))}</td>
                  <td><strong>${escapeHTML(row.name)}</strong></td>
                  <td>${escapeHTML(row.team || "—")}</td>
                  <td>${escapeHTML(row.position || "—")}</td>
                  <td class="playoff-archetype-column">${profileChip(row.regProfile)}</td>
                  <td class="playoff-archetype-column">${profileChip(row.playoffProfile)}</td>
                  <td class="numeric">${number(row.regGames)}</td>
                  <td class="numeric">${number(row.playoffGames)}</td>
                  <td class="numeric">${number(row.regPpg, 3)}</td>
                  <td class="numeric">${number(row.playoffPpg, 3)}</td>
                  <td class="numeric">${signedNumber(row.ppgChange, 2)}</td>
                  <td class="numeric">${signedNumber(row.toiChange, 1)}</td>
                  <td class="numeric">${playoffScorePill(row.distance)}</td>
                  <td>${playoffBandPill(row.shiftBand)}</td>
                  <td>${escapeHTML(row.sampleReliability || "unknown")}</td>
                  <td class="numeric">${number(row.statShift, 2)}</td>
                  <td>${playoffBandPill(row.statBand)}</td>
                  <td>${minuteClock(row.regToi)}</td>
                  <td>${minuteClock(row.playoffToi)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function playoffScatter(rows) {
  const xValues = rows.map((row) => Number(row.ppgChange || 0));
  const yValues = rows.map((row) => Number(row.toiChange || 0));
  const xMin = Math.min(0, ...xValues);
  const xMax = Math.max(0, ...xValues);
  const yMin = Math.min(0, ...yValues);
  const yMax = Math.max(0, ...yValues);
  const xPadding = Math.max((xMax - xMin) * 0.08, 0.05);
  const yPadding = Math.max((yMax - yMin) * 0.08, 0.4);
  const boundedXMin = xMin - xPadding;
  const boundedXMax = xMax + xPadding;
  const boundedYMin = yMin - yPadding;
  const boundedYMax = yMax + yPadding;
  const xRange = Math.max(boundedXMax - boundedXMin, 0.01);
  const yRange = Math.max(boundedYMax - boundedYMin, 0.01);
  const xPosition = (value) =>
    ((Number(value || 0) - boundedXMin) / xRange) * 100;
  const yPosition = (value) =>
    ((Number(value || 0) - boundedYMin) / yRange) * 100;
  const profiles = [...new Set(rows.map((row) => row.regProfile))];
  return `
    <section class="playoff-scatter-card" aria-labelledby="playoff-scatter-title">
      <header>
        <div>
          <p class="eyebrow">Scatter plot</p>
          <h3 id="playoff-scatter-title">Scoring and ice-time change</h3>
        </div>
        <p>
          Each dot is one player. X-axis = scoring rate change (playoff P/GP
          minus regular-season P/GP). Y-axis = ice-time change (playoff ATOI
          minus regular-season ATOI). <strong>Dot shape</strong> encodes the
          model shift band (how much the archetype fingerprint moved);
          <strong>dot size</strong> encodes playoff games played;
          <strong>color</strong> encodes regular-season archetype.
        </p>
      </header>
      <div class="playoff-scatter-stage">
        <span class="playoff-scatter-y-label">Playoff ATOI − Regular-season ATOI (min)</span>
        <div class="playoff-scatter-plot">
          <span class="playoff-scatter-zero-x" style="left:${xPosition(0)}%"></span>
          <span class="playoff-scatter-zero-y" style="bottom:${yPosition(0)}%"></span>
          ${rows
            .map((row) => {
              const size = Math.min(
                30,
                10 + Math.sqrt(Number(row.playoffGames || 0)) * 3,
              );
              const tooltip = `${row.name} · ${row.team || "—"} · REG: ${row.regProfile} · PO: ${row.playoffProfile} · Model shift ${number(row.distance, 3)} · P/GP ${signedNumber(row.ppgChange, 2)} · ATOI ${signedNumber(row.toiChange, 1)} min · ${number(row.playoffGames)} PO GP`;
              const shape = {
                "Held steady": "is-circle",
                "Moderate shift": "is-square",
                "Major shift": "is-triangle",
              }[row.shiftBand] || "is-cross";
              return `
                <button
                  class="playoff-scatter-point ${shape}"
                  type="button"
                  style="left:${xPosition(row.ppgChange)}%;bottom:${yPosition(row.toiChange)}%;--point-size:${size}px;--profile:${profileColor(row.regProfile)}"
                  data-tooltip="${escapeHTML(tooltip)}"
                  aria-label="${escapeHTML(tooltip)}"
                ></button>
              `;
            })
            .join("")}
        </div>
        <span class="playoff-scatter-x-label">Playoff P/GP − Regular-season P/GP</span>
      </div>
      <div class="playoff-scatter-key">
        <div class="playoff-shape-key" aria-label="Model shift band shapes">
          <span><i class="is-circle"></i>Held steady</span>
          <span><i class="is-square"></i>Moderate shift</span>
          <span><i class="is-triangle"></i>Major shift</span>
        </div>
        <div class="playoff-profile-key" aria-label="Regular-season archetypes">
          ${profiles
            .map(
              (profile) => `
                <span>
                  <i style="--profile:${profileColor(profile)}"></i>
                  ${escapeHTML(profile)}
                </span>
              `,
            )
            .join("")}
        </div>
      </div>
    </section>
  `;
}

function playoffSeasonView(rows) {
  if (!rows.length) {
    return '<div class="empty-state">No players match the current filters.</div>';
  }
  const changed = rows.filter((row) => row.changed).length;
  return `
    <section class="playoff-section-head">
      <p class="eyebrow">Season View</p>
      <h2>${escapeHTML(playoffSeasonLabel(appState.playoffSeason))} Playoff Shifts</h2>
    </section>
    <section class="metric-grid playoff-metric-grid">
      ${metric("Players", number(rows.length))}
      ${metric("Median model shift", number(median(rows.map((row) => row.distance)), 2))}
      ${metric("Archetype changes", number(changed))}
      ${metric("% changed archetype", percent(rows.length ? changed / rows.length : 0))}
    </section>
    ${playoffScatter(rows)}
    <section class="playoff-profile-changes">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Model shift score</p>
          <h2>Biggest Playoff Profile Changes (sorted by model shift score)</h2>
        </div>
      </div>
      ${playoffShiftTable(rows)}
    </section>
  `;
}

function playoffArchetypeRows(base) {
  const groups = new Map();
  base.forEach((row) => {
    const key = `${row.season}|||${row.regProfile}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  });
  return [...groups.entries()]
    .map(([key, rows]) => {
      const [season, profile] = key.split("|||");
      const changed = rows.filter((row) => row.changed).length;
      return {
        season,
        profile,
        players: new Set(rows.map((row) => row.id)).size,
        medianModelShift: median(rows.map((row) => row.distance)),
        changeRate: rows.length ? changed / rows.length : 0,
        medianStatShift: median(rows.map((row) => row.statShift)),
        medianPpgChange: median(rows.map((row) => row.ppgChange)),
        medianToiChange: median(rows.map((row) => row.toiChange)),
      };
    })
    .filter((row) => row.players >= 3);
}

function playoffArchetypeMatrix(rows) {
  const seasons = [...new Set(rows.map((row) => row.season))].sort();
  const profiles = [...new Set(rows.map((row) => row.profile))].sort(
    (left, right) =>
      median(
        rows
          .filter((row) => row.profile === right)
          .map((row) => row.medianModelShift),
      ) -
        median(
          rows
            .filter((row) => row.profile === left)
            .map((row) => row.medianModelShift),
        ) ||
      left.localeCompare(right),
  );
  const cells = new Map(
    rows.map((row) => [`${row.profile}|||${row.season}`, row]),
  );
  return `
    <div class="playoff-matrix-scroll" tabindex="0" aria-label="Archetype shift matrix; scroll horizontally for all seasons">
      <div
        class="playoff-matrix"
        style="--season-count:${seasons.length};--matrix-width:${Math.max(860, 270 + seasons.length * 68)}px"
      >
        <span class="playoff-matrix-corner">REG archetype</span>
        ${seasons
          .map(
            (season) => `
              <span class="playoff-matrix-season">${escapeHTML(careerCompactSeasonLabel(season))}</span>
            `,
          )
          .join("")}
        ${profiles
          .map(
            (profile) => `
              <strong class="playoff-matrix-profile">${escapeHTML(profile)}</strong>
              ${seasons
                .map((season) => {
                  const row = cells.get(`${profile}|||${season}`);
                  if (!row) {
                    return '<span class="playoff-matrix-cell is-empty"></span>';
                  }
                  const size = 10 + row.changeRate * 28;
                  const alpha = Math.max(
                    0.14,
                    Math.min(1, row.medianModelShift),
                  );
                  const label = `${playoffSeasonLabel(season)} · ${profile} · ${row.players} players · Median model shift ${number(row.medianModelShift, 3)} · Archetype change rate ${percent(row.changeRate)}`;
                  return `
                    <span
                      class="playoff-matrix-cell"
                      title="${escapeHTML(label)}"
                    >
                      <i
                        style="--dot-size:${size}px;--dot-alpha:${alpha}"
                        role="img"
                        aria-label="${escapeHTML(label)}"
                      ></i>
                    </span>
                  `;
                })
                .join("")}
            `,
          )
          .join("")}
      </div>
    </div>
  `;
}

function playoffArchetypeTable(rows) {
  const sorted = [...rows].sort(
    (left, right) =>
      right.season.localeCompare(left.season) ||
      right.medianModelShift - left.medianModelShift,
  );
  return `
    <div class="playoff-table-wrap" tabindex="0" aria-label="Archetype playoff shift table; scroll horizontally for all columns">
      <table class="playoff-table">
        <thead>
          <tr>
            <th>Season</th>
            <th class="playoff-archetype-column">REG archetype</th>
            <th class="numeric">Players</th>
            <th class="numeric">Median model shift</th>
            <th class="numeric">Archetype change rate</th>
            <th class="numeric">Median stat shift</th>
            <th class="numeric">Median P/GP change</th>
            <th class="numeric">Median TOI change</th>
          </tr>
        </thead>
        <tbody>
          ${sorted
            .map(
              (row) => `
                <tr>
                  <td>${escapeHTML(playoffSeasonLabel(row.season))}</td>
                  <td class="playoff-archetype-column">${profileChip(row.profile)}</td>
                  <td class="numeric">${number(row.players)}</td>
                  <td class="numeric">${number(row.medianModelShift, 3)}</td>
                  <td class="numeric">${percent(row.changeRate)}</td>
                  <td class="numeric">${number(row.medianStatShift, 2)}</td>
                  <td class="numeric">${signedNumber(row.medianPpgChange, 2)}</td>
                  <td class="numeric">${signedNumber(row.medianToiChange, 1)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function playoffArchetypesView(base) {
  const rows = playoffArchetypeRows(base);
  if (!rows.length) {
    return '<div class="empty-state">No archetypes match the current filters.</div>';
  }
  return `
    <section class="playoff-section-head">
      <p class="eyebrow">Archetypes</p>
      <h2>Regular-Season Archetypes Under Playoff Pressure</h2>
      <p>
        For each regular-season archetype and season, the chart shows how much
        that group's play style shifted in the playoffs — and how many players
        actually got re-classified into a different archetype. Hover over any
        circle for full detail.
      </p>
    </section>
    <aside class="playoff-how-to-read">
      <strong>How to read this:</strong>
      <span><i class="playoff-color-swatch"></i><strong>Color</strong> (light→dark orange-red) = median model shift score — darker means players looked more different in the playoffs.</span>
      <span><i class="playoff-size-swatch"></i><strong>Size</strong> = archetype change rate — bigger dot means a higher share of players got classified into a <em>different</em> archetype in the playoffs vs regular season.</span>
    </aside>
    ${playoffArchetypeMatrix(rows)}
    ${playoffArchetypeTable(rows)}
  `;
}

function playoffEligiblePlayers(base) {
  const grouped = new Map();
  [...base]
    .sort((left, right) => left.season.localeCompare(right.season))
    .forEach((row) => {
      if (!grouped.has(row.id)) {
        grouped.set(row.id, {
          id: row.id,
          name: row.name,
          position: row.position,
          team: row.team,
          firstSeason: row.season,
          lastSeason: row.season,
          latestSeason: row.season,
          seasons: 0,
        });
      }
      const player = grouped.get(row.id);
      player.seasons += 1;
      if (row.season < player.firstSeason) player.firstSeason = row.season;
      if (row.season > player.lastSeason) player.lastSeason = row.season;
      if (row.season >= player.latestSeason) {
        player.latestSeason = row.season;
        player.position = row.position;
        player.team = row.team;
      }
    });
  return [...grouped.values()]
    .map((row) => ({
      ...row,
      display: `${row.name} — ${row.position || "UNK"} — ${playoffSeasonLabel(row.firstSeason)} to ${playoffSeasonLabel(row.lastSeason)}`,
    }))
    .sort(
      (left, right) =>
        right.latestSeason.localeCompare(left.latestSeason) ||
        left.name.localeCompare(right.name),
    );
}

function playoffCareerStatistics(history) {
  if (!history.length) return "";
  const playoffStyleCounts = new Map();
  const playoffStyleRecency = new Map();
  history.forEach((row, index) => {
    playoffStyleCounts.set(
      row.playoffProfile,
      (playoffStyleCounts.get(row.playoffProfile) || 0) + 1,
    );
    playoffStyleRecency.set(row.playoffProfile, index);
  });
  const mostFrequentPlayoffStyle =
    [...playoffStyleCounts.keys()].sort(
      (left, right) =>
        playoffStyleCounts.get(right) - playoffStyleCounts.get(left) ||
        playoffStyleRecency.get(right) - playoffStyleRecency.get(left),
    )[0] || "—";
  const total = (field) =>
    history.reduce((sum, row) => sum + Number(row[field] || 0), 0);
  const average = (field) => mean(history.map((row) => Number(row[field] || 0)));
  const classificationChanges = history.filter((row) => row.changed).length;

  return `
    <section
      class="career-card-summary playoff-career-card-summary"
      aria-labelledby="playoff-career-card-summary-title"
    >
      <header>
        <p class="career-card-eyebrow" id="playoff-career-card-summary-title">
          Career totals in dataset
        </p>
      </header>
      <div class="career-card-stat-groups">
        <section class="career-card-stat-group" aria-labelledby="playoff-career-reg-title">
          <h3 id="playoff-career-reg-title">Regular season sample</h3>
          <dl class="career-card-stat-grid">
            ${careerCardStat("Seasons", number(history.length))}
            ${careerCardStat("GP", number(total("regGames")))}
            ${careerCardStat("P/GP", number(average("regPpg"), 2))}
            ${careerCardStat("ATOI", minuteClock(average("regToi")))}
          </dl>
        </section>
        <section class="career-card-stat-group" aria-labelledby="playoff-career-po-title">
          <h3 id="playoff-career-po-title">Playoffs</h3>
          <dl class="career-card-stat-grid">
            ${careerCardStat("Seasons", number(history.length))}
            ${careerCardStat("GP", number(total("playoffGames")))}
            ${careerCardStat("P/GP", number(average("playoffPpg"), 2))}
            ${careerCardStat("ATOI", minuteClock(average("playoffToi")))}
          </dl>
        </section>
      </div>
      <div class="career-card-style">
        <div class="career-card-primary-style">
          <span>Most frequent playoff style</span>
          ${profileChip(mostFrequentPlayoffStyle)}
        </div>
        <div class="career-card-change-count">
          <span>REG → PO changes</span>
          <strong>${number(classificationChanges)}</strong>
        </div>
      </div>
    </section>
  `;
}

function playoffCareerSummary(history) {
  const medianModelShift = median(history.map((row) => row.distance));
  const averagePpgChange = mean(history.map((row) => row.ppgChange));
  return `
    <section
      class="career-summary-grid playoff-career-summary-grid"
      aria-label="Player career playoff summary"
    >
      <article class="career-summary-card">
        <span>Playoff seasons</span>
        <strong>${number(history.length)}</strong>
      </article>
      <article class="career-summary-card">
        <span>Median model shift</span>
        <div>
          <strong>${number(medianModelShift, 2)}</strong>
          ${playoffBandPill(
            medianModelShift > 0.75
              ? "Major shift"
              : medianModelShift > 0.25
                ? "Moderate shift"
                : "Held steady",
          )}
        </div>
      </article>
      <article class="career-summary-card">
        <span>Career PO GP</span>
        <strong>${number(
          history.reduce(
            (sum, row) => sum + Number(row.playoffGames || 0),
            0,
          ),
        )}</strong>
      </article>
      <article class="career-summary-card">
        <span>Career P/GP change</span>
        <div>
          <strong>${signedNumber(averagePpgChange, 2)}</strong>
          <em class="playoff-direction ${averagePpgChange >= 0 ? "is-up" : "is-down"}">
            ${averagePpgChange >= 0 ? "Higher in playoffs" : "Lower in playoffs"}
          </em>
        </div>
      </article>
    </section>
  `;
}

function playoffDeltaChart(history, regField, playoffField, title, digits) {
  const values = history.flatMap((row) => [
    Number(row[regField] || 0),
    Number(row[playoffField] || 0),
  ]);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const padding = Math.max((maxValue - minValue) * 0.08, 0.05);
  const lower = minValue - padding;
  const range = Math.max(maxValue + padding - lower, 0.01);
  const position = (value) =>
    ((Number(value || 0) - lower) / range) * 100;
  return `
    <section class="playoff-delta-card">
      <header>
        <h3>${escapeHTML(title)}</h3>
        <span><i class="is-regular"></i>Regular Season</span>
        <span><i class="is-playoffs"></i>Playoffs</span>
      </header>
      <div class="playoff-delta-list">
        ${history
          .map((row) => {
            const regPosition = position(row[regField]);
            const poPosition = position(row[playoffField]);
            const start = Math.min(regPosition, poPosition);
            const width = Math.abs(poPosition - regPosition);
            return `
              <div class="playoff-delta-row">
                <span>${escapeHTML(playoffSeasonLabel(row.season))}</span>
                <div class="playoff-delta-track">
                  <i class="playoff-delta-line" style="left:${start}%;width:${width}%"></i>
                  <i class="playoff-delta-point is-regular" style="left:${regPosition}%"></i>
                  <i class="playoff-delta-point is-playoffs" style="left:${poPosition}%"></i>
                </div>
                <span>${number(row[regField], digits)} → ${number(row[playoffField], digits)}</span>
              </div>
            `;
          })
          .join("")}
      </div>
    </section>
  `;
}

function playoffCareerTranslation(history) {
  const maxShift = Math.max(1, ...history.map((row) => Number(row.distance || 0)));
  return `
    <div class="playoff-career-comparison">
      <section class="playoff-translation-card">
        <h3>Archetype Translation</h3>
        <div class="playoff-translation-head">
          <span>Season</span><span>Regular Season</span><span>Playoffs</span>
        </div>
        ${history
          .map(
            (row) => `
              <div class="playoff-translation-row">
                <strong>${escapeHTML(playoffSeasonLabel(row.season))}</strong>
                ${profileChip(row.regProfile)}
                ${profileChip(row.playoffProfile)}
              </div>
            `,
          )
          .join("")}
      </section>
      <section class="playoff-shift-card">
        <h3>How Much the Playoff Profile Moved</h3>
        <div class="playoff-shift-bars">
          ${history
            .map(
              (row) => `
                <div class="playoff-shift-row">
                  <span>${escapeHTML(playoffSeasonLabel(row.season))}</span>
                  <div class="playoff-shift-track">
                    <i style="width:${(Number(row.distance || 0) / maxShift) * 100}%"></i>
                  </div>
                  <strong>${number(row.distance, 3)}</strong>
                </div>
              `,
            )
            .join("")}
        </div>
      </section>
    </div>
  `;
}

function playoffCareerTrajectory(history) {
  return `
    <section
      class="career-timeline-panel playoff-career-trajectory"
      aria-labelledby="playoff-career-trajectory-title"
    >
      <header class="career-timeline-head">
        <h2 id="playoff-career-trajectory-title">Regular Season → Playoffs</h2>
        <ul>
          <li>Each row connects the player's regular-season result to their playoff result in the same year.</li>
          <li>Longer connectors show a larger change in scoring rate or average ice time.</li>
        </ul>
      </header>
      <div class="playoff-career-trajectory-body">
        <div class="playoff-delta-grid">
          ${playoffDeltaChart(
            history,
            "regPpg",
            "playoffPpg",
            "Scoring Rate: Regular Season vs Playoffs",
            2,
          )}
          ${playoffDeltaChart(
            history,
            "regToi",
            "playoffToi",
            "Ice Time: Regular Season vs Playoffs (min)",
            1,
          )}
        </div>
        ${playoffCareerTranslation(history)}
      </div>
    </section>
  `;
}

function playoffCareerSeasonCard(row, index, total) {
  const isLatest = index === total - 1;
  return `
    <li class="career-season-item">
      <details
        class="career-season-card playoff-career-season-card"
        id="playoff-career-season-${escapeHTML(row.season)}"
        style="--profile:${profileColor(row.playoffProfile)}"
        ${isLatest ? "open" : ""}
      >
        <summary>
          <span class="career-season-node" aria-hidden="true"></span>
          <span class="career-season-summary">
            <span class="career-season-identity">
              <span class="career-season-title-row">
                <span class="career-field-label">Season</span>
                <strong>${escapeHTML(playoffSeasonLabel(row.season))}</strong>
                ${row.changed ? '<span class="career-change-label">REG → PO change</span>' : ""}
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
            <span class="career-season-profile playoff-season-profile">
              <span>Regular season → playoffs</span>
              <span class="playoff-profile-transition">
                ${profileChip(row.regProfile)}
                <i aria-hidden="true">→</i>
                ${profileChip(row.playoffProfile)}
              </span>
            </span>
            <span class="career-season-measures">
              <span>
                <small>Model shift</small>
                <strong>${number(row.distance, 3)}</strong>
              </span>
              <span>
                <small>Stat shift</small>
                <strong>${number(row.statShift, 2)}</strong>
              </span>
            </span>
            <span class="career-season-disclosure" aria-hidden="true"></span>
          </span>
        </summary>
        <div class="career-season-detail playoff-career-season-detail">
          <section>
            <h3>Regular season</h3>
            <dl class="career-stat-grid">
              ${careerStat("REG GP", number(row.regGames))}
              ${careerStat("REG P/GP", number(row.regPpg, 3))}
              ${careerStat("REG ATOI", minuteClock(row.regToi))}
              ${careerStat("Style", row.regProfile)}
            </dl>
          </section>
          <section>
            <h3>Playoffs</h3>
            <dl class="career-stat-grid">
              ${careerStat("PO GP", number(row.playoffGames))}
              ${careerStat("PO P/GP", number(row.playoffPpg, 3))}
              ${careerStat("PO ATOI", minuteClock(row.playoffToi))}
              ${careerStat("Style", row.playoffProfile)}
            </dl>
          </section>
          <section>
            <h3>What changed</h3>
            <dl class="career-stat-grid">
              ${careerStat("P/GP", signedNumber(row.ppgChange, 2))}
              ${careerStat("ATOI", `${signedNumber(row.toiChange, 1)} min`)}
              ${careerStat("Shot rate", signedNumber(row.shotRateChange, 2))}
              ${careerStat("PIM rate", signedNumber(row.pimRateChange, 2))}
              ${careerStat("+/− rate", signedNumber(row.plusMinusRateChange, 2))}
              ${careerStat("Model shift", number(row.distance, 3))}
              ${careerStat("Stat shift", number(row.statShift, 2))}
              ${careerStat("Shift band", row.shiftBand || "—")}
            </dl>
          </section>
        </div>
      </details>
    </li>
  `;
}

function playoffCareerSeasonList(history) {
  return `
    <section
      class="career-season-section playoff-career-season-section"
      aria-labelledby="playoff-career-season-title"
    >
      <h2 id="playoff-career-season-title">Playoff Translation by Season</h2>
      <ol class="career-season-list">
        ${history
          .map((row, index) =>
            playoffCareerSeasonCard(row, index, history.length),
          )
          .join("")}
      </ol>
    </section>
  `;
}

function playoffPlayerView(base) {
  const players = playoffEligiblePlayers(base);
  if (
    !appState.playoffPlayerId ||
    !players.some((player) => player.id === appState.playoffPlayerId)
  ) {
    appState.playoffPlayerId = players[0]?.id;
  }
  const history = base
    .filter((row) => row.id === appState.playoffPlayerId)
    .sort((left, right) => left.season.localeCompare(right.season));
  const selected = players.find(
    (player) => player.id === appState.playoffPlayerId,
  );
  const latest = history.at(-1);
  const selectedWithTeam = selected
    ? {
        ...selected,
        team: latest?.team || selected.team,
        position: latest?.position || selected.position,
      }
    : null;
  if (!players.length || !history.length) {
    return '<div class="empty-state">No matching playoff players under the current filters.</div>';
  }
  return `
    <section class="playoff-section-head">
      <p class="eyebrow">Player Career</p>
      <h2>Player Career Playoff Pattern</h2>
      <p>
        Follow how one player's regular-season identity translated under
        playoff pressure, season by season.
      </p>
    </section>
    <section
      class="career-picker playoff-player-picker"
      aria-labelledby="playoff-player-picker-title"
    >
      <div class="career-picker-controls playoff-player-controls">
        <h2 id="playoff-player-picker-title">Select a player</h2>
        <div class="field">
          <label for="playoff-player-search">Search player name</label>
          <input
            class="search-input"
            id="playoff-player-search"
            type="search"
            value="${escapeHTML(appState.playoffQuery)}"
            autocomplete="off"
            aria-controls="playoff-player-matches"
          />
        </div>
        <div class="field">
          <label for="playoff-player-matches">Matches</label>
          <select id="playoff-player-matches" size="6">
            ${players
              .map(
                (player) => `
                  <option
                    value="${player.id}"
                    ${player.id === appState.playoffPlayerId ? "selected" : ""}
                  >${escapeHTML(player.display)}</option>
                `,
              )
              .join("")}
          </select>
        </div>
        <p class="playoff-player-match-status" aria-live="polite"></p>
        <p class="playoff-player-no-matches" hidden>No matching playoff players under the current filters.</p>
        <p class="career-selected-caption">
          Selected: ${escapeHTML(selected?.display || "No player")}
        </p>
      </div>
      <div
        class="detail-panel career-selected-player playoff-selected-player"
        tabindex="-1"
        aria-label="${escapeHTML(`Selected player: ${selected?.name || "No player"}`)}"
      >
        ${playerIdentity(
          selectedWithTeam,
          latest?.season || selected?.latestSeason,
          "Selected player",
          `${selectedWithTeam.position || "UNK"}
${playoffSeasonLabel(selected?.firstSeason)} to ${playoffSeasonLabel(selected?.lastSeason)}`,
        )}
        ${playoffCareerStatistics(history)}
      </div>
    </section>
    ${playoffCareerSummary(history)}
    ${playoffCareerTrajectory(history)}
    ${playoffCareerSeasonList(history)}
    <section
      class="playoff-career-table-section"
      aria-labelledby="playoff-career-table-title"
    >
      <h2 id="playoff-career-table-title">Complete Career Comparison</h2>
      <p>Every regular-season and playoff value used in the career view.</p>
    </section>
    ${playoffShiftTable(history, history.length, "season")}
  `;
}

async function renderPlayoffs() {
  if (!appState.playoffs) {
    loading("Loading playoff histories…");
    appState.playoffs = await getJSON(
      `${DATA_ROOT}/playoffs.json?v=${DATA_VERSION}`,
    );
    const viable = [
      ...new Set(
        appState.playoffs
          .filter((row) => row.playoffGames >= 1)
          .map((row) => row.season),
      ),
    ]
      .sort()
      .reverse();
    appState.playoffSeason = viable[0] || appState.playoffs[0]?.season;
  }
  if (appState.route !== "playoffs") return;
  cleanupCanvases();
  const base = playoffBaseRows();
  const seasonRows = playoffSeasonRows(base);
  const playoffRegMax = Number(
    appState.core?.meta?.scheduleDimension?.[appState.playoffSeason] || 82,
  );
  main.innerHTML = `
    <article class="page playoff-page">
      <header class="playoff-page-head">
        <div>
          <h1>How Does Play Style Change in the Playoffs?</h1>
          <p>
            We all know that the playoffs feel different — tighter systems,
            better goaltending, and higher stakes. But how much does a player's
            <em>actual style</em> change when the intensity ramps up? This page
            answers that question using the same archetype model that classifies
            regular-season play, now applied to playoff data.
          </p>
        </div>
        <div class="controls">${playoffControls()}</div>
      </header>
      ${playoffExplainer()}
      ${playoffPrimarySignal()}
      <section class="playoff-filter-panel" aria-label="Playoff sample filters">
        <div class="field">
          <label for="playoff-reg-games">
            Min regular-season games
            <output id="playoff-reg-value">${appState.playoffMinReg}</output>
          </label>
          <input
            id="playoff-reg-games"
            type="range"
            min="0"
            max="${playoffRegMax}"
            step="5"
            value="${appState.playoffMinReg}"
          />
          <span><small>0</small><small>${playoffRegMax}</small></span>
        </div>
        <div class="field">
          <label for="playoff-po-games">
            Min playoff games
            <output id="playoff-po-value">${appState.playoffMinPo}</output>
          </label>
          <input
            id="playoff-po-games"
            type="range"
            min="1"
            max="28"
            step="1"
            value="${appState.playoffMinPo}"
          />
          <span><small>1</small><small>28</small></span>
        </div>
      </section>
      ${tabs(
        [
          ["season", "Season View"],
          ["archetypes", "Archetypes"],
          ["player", "Player Career"],
        ],
        appState.playoffTab,
        "playoffs",
      )}
      <section id="playoff-panel">
        ${
          appState.playoffTab === "season"
            ? playoffSeasonView(seasonRows)
            : appState.playoffTab === "archetypes"
              ? playoffArchetypesView(base)
              : playoffPlayerView(base)
        }
      </section>
    </article>
  `;
  hydratePlayerAssets(main);

  bindGroupControl("playoff-group", (group) => {
    appState.playoffGroup = group;
    appState.playoffPlayerId = null;
    appState.playoffQuery = "";
    renderPlayoffs();
  });
  document
    .querySelector("#playoff-season")
    ?.addEventListener("change", (event) => {
      appState.playoffSeason = event.target.value;
      renderPlayoffs();
    });
  document
    .querySelector("#playoff-reg-games")
    ?.addEventListener("input", (event) => {
      document.querySelector("#playoff-reg-value").textContent =
        event.target.value;
    });
  document
    .querySelector("#playoff-reg-games")
    ?.addEventListener("change", (event) => {
      appState.playoffMinReg = Number(event.target.value);
      appState.playoffPlayerId = null;
      renderPlayoffs();
    });
  document
    .querySelector("#playoff-po-games")
    ?.addEventListener("input", (event) => {
      document.querySelector("#playoff-po-value").textContent =
        event.target.value;
    });
  document
    .querySelector("#playoff-po-games")
    ?.addEventListener("change", (event) => {
      appState.playoffMinPo = Number(event.target.value);
      appState.playoffPlayerId = null;
      renderPlayoffs();
    });
  bindTabs("playoffs", (tab) => {
    appState.playoffTab = tab;
    renderPlayoffs();
  });
  if (appState.playoffTab === "player") {
    bindPlayoffPlayerSearch(base);
  }
}

function bindPlayoffPlayerSearch(base) {
  const players = playoffEligiblePlayers(base);
  const search = document.querySelector("#playoff-player-search");
  const matches = document.querySelector("#playoff-player-matches");
  const status = document.querySelector(".playoff-player-match-status");
  const noMatches = document.querySelector(".playoff-player-no-matches");
  const draw = () => {
    if (!search || !matches) return;
    appState.playoffQuery = search.value;
    const query = search.value.trim().toLowerCase();
    const filtered = players.filter(
      (player) => !query || player.name.toLowerCase().includes(query),
    );
    matches.innerHTML = filtered
      .map(
        (player) => `
          <option
            value="${player.id}"
            ${player.id === appState.playoffPlayerId ? "selected" : ""}
          >${escapeHTML(player.display)}</option>
        `,
      )
      .join("");
    matches.disabled = filtered.length === 0;
    if (status) {
      status.textContent = `${number(filtered.length)} matching player${filtered.length === 1 ? "" : "s"}`;
    }
    if (noMatches) noMatches.hidden = filtered.length > 0;
  };
  search?.addEventListener("input", draw);
  matches?.addEventListener("change", () => {
    const playerId = Number(matches.value);
    if (!playerId) return;
    appState.playoffPlayerId = playerId;
    appState.playoffQuery = "";
    renderPlayoffs().then(() => {
      document
        .querySelector(".playoff-selected-player")
        ?.focus({ preventScroll: true });
    });
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
      `${latest} complete · PO through ${appState.core.meta.playoffDataThrough || "—"}`;
    await renderRoute();
  } catch (error) {
    showError(error);
  }
}

window.addEventListener("hashchange", renderRoute);
init();
