const DATA_ROOT = "/data";
const DATA_VERSION = "20260729-season-reads-v1";

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

function playerVisual(player, season) {
  if (!player) return "";
  const teams = teamCodes(player.team);
  const seasonKey = /^\d{8}$/.test(String(season)) ? String(season) : "latest";
  const playerId = String(player.id || "").replace(/\D/g, "");
  const headshotSources = playerId
    ? [
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
      ]
    : [];
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
            .map((paragraph) => `<p>${escapeHTML(paragraph)}</p>`)
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
  const latest = [...history].sort((a, b) => b.season.localeCompare(a.season))[0];
  const selectedWithTeam = selected
    ? {
        ...selected,
        team: latest?.team || "",
        position: latest?.position || selected.position,
      }
    : null;

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
          ${
            selectedWithTeam
              ? playerIdentity(
                  selectedWithTeam,
                  latest?.season,
                  "Selected player",
                  `${selectedWithTeam.position} · ${number(selected?.seasons || 0)} seasons`,
                )
              : '<div class="detail-name">No player</div>'
          }
        </div>
      </div>
      <section id="career-view">${careerView(history)}</section>
      <footer class="page-footer">
        <span>Styles are re-learned each season.</span>
        <span>A switch means the top probability changed.</span>
      </footer>
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
