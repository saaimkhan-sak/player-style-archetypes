import { access, readFile } from "node:fs/promises";

const required = [
  "index.html",
  "styles.css",
  "app.js",
  "data/core.json",
  "data/careers.json",
  "data/playoffs.json",
];

await Promise.all(required.map((path) => access(path)));

const [coreSource, appSource, stylesSource, careersSource, indexSource] = await Promise.all([
  readFile("data/core.json", "utf8"),
  readFile("app.js", "utf8"),
  readFile("styles.css", "utf8"),
  readFile("data/careers.json", "utf8"),
  readFile("index.html", "utf8"),
]);
const data = JSON.parse(coreSource);
const careers = JSON.parse(careersSource);
if (!data.meta?.seasons?.length || !data.glossary) {
  throw new Error("The generated site data is incomplete.");
}
if (
  !/\.bar-fill\s*\{[\s\S]*?display:\s*block;[\s\S]*?\}/.test(
    stylesSource,
  ) ||
  appSource.includes("profile.share / maxShare")
) {
  throw new Error("Season profile bars are not using visible, literal shares.");
}
if (
  !appSource.includes("https://assets.nhle.com/mugs/nhl/") ||
  !appSource.includes("https://assets.nhle.com/logos/nhl/svg/") ||
  !appSource.includes("<span>Games played</span>") ||
  !/\.player-headshot\s*\{[\s\S]*?object-fit:\s*cover;[\s\S]*?\}/.test(
    stylesSource,
  ) ||
  !stylesSource.includes(
    ".player-headshot-frame.is-loaded .player-headshot-initials",
  ) ||
  !/\.team-logo-list\s*\{[\s\S]*?position:\s*static;[\s\S]*?grid-area:\s*logos;[\s\S]*?\}/.test(
    stylesSource,
  ) ||
  !/\.player-identity\s*\{[\s\S]*?grid-template-areas:\s*"portrait copy logos";[\s\S]*?\}/.test(
    stylesSource,
  ) ||
  !/\.team-logo-list\s*\{[\s\S]*?align-self:\s*start;[\s\S]*?justify-self:\s*end;[\s\S]*?\}/.test(
    stylesSource,
  ) ||
  !/\.team-logo-frame\s*\{[\s\S]*?width:\s*72px;[\s\S]*?height:\s*72px;[\s\S]*?border:\s*0;[\s\S]*?border-radius:\s*0;[\s\S]*?background:\s*transparent;[\s\S]*?box-shadow:\s*none;[\s\S]*?\}/.test(
    stylesSource,
  ) ||
  !/\.team-logo\s*\{[\s\S]*?inset:\s*0;[\s\S]*?width:\s*100%;[\s\S]*?height:\s*100%;[\s\S]*?\}/.test(
    stylesSource,
  )
) {
  throw new Error(
    "Player profile cards are missing their corrected headshot, top-right team logo, or games-played treatment.",
  );
}
if (
  !appSource.includes("function profileMentionMarkup(value, profileNames)") ||
  !appSource.includes("escapeHTML(text.slice(offset, index))") ||
  !appSource.includes(">${escapeHTML(name)}</strong>") ||
  !appSource.includes("profileMentionMarkup(paragraph, profileNames)") ||
  !/\.season-read-profile\s*\{[\s\S]*?var\(--profile\)[\s\S]*?\}/.test(
    stylesSource,
  )
) {
  throw new Error(
    "Season reads are missing safe, color-coded archetype mentions.",
  );
}
if (
  !appSource.includes("<h2>Team Roster Construction</h2>") ||
  !appSource.includes(
    "A depth-chart view of the selected team using the 12 forwards or 8 defensemen with the most regular-season ice time.",
  ) ||
  !appSource.includes(
    "Whether the team identity is concentrated in stars or spread through depth.",
  ) ||
  !appSource.includes("function rosterPlayerHeadshot(player, season)") ||
  !appSource.includes("function rosterTeamLogo(team, season)") ||
  !appSource.includes("hydratePlayerAssets(teamViewElement)") ||
  !stylesSource.includes(".roster-unit-grid") ||
  !stylesSource.includes(".roster-player-photo-image") ||
  !stylesSource.includes(".roster-team-logo-image")
) {
  throw new Error(
    "Team roster construction is missing its Streamlit content or visual assets.",
  );
}
if (
  !appSource.includes(
    "<h2>Need Finder (find players who match a target archetype)</h2>",
  ) ||
  !appSource.includes(
    "A ranked list of players who best match a selected style profile.",
  ) ||
  !appSource.includes(
    "“Target similarity (%)” is the model’s estimated probability that the player belongs to that archetype.",
  ) ||
  !appSource.includes("<label for=\"need-team\">Exclude team (optional)</label>") ||
  !appSource.includes("<option value=\"\">(none)</option>") ||
  !appSource.includes("<label for=\"need-profile\">Target archetype</label>") ||
  !appSource.includes("<label for=\"need-games\">Min REG games</label>") ||
  !appSource.includes("const NEED_GAME_VALUES = [") ||
  !appSource.includes("80, 82,") ||
  !appSource.includes("aria-valuemax=\"82\"") ||
  !appSource.includes("const minGames = needGamesValue(games)") ||
  !appSource.includes("const matches = eligible.slice(0, 80)") ||
  !appSource.includes("Number(b.points || 0) - Number(a.points || 0)") ||
  !appSource.includes("hydratePlayerAssets(results)") ||
  !stylesSource.includes(".need-finder-intro") ||
  !stylesSource.includes(".need-player-detail") ||
  !stylesSource.includes(".need-similarity-track")
) {
  throw new Error(
    "Need Finder is missing its Streamlit content, exact ranking logic, or responsive scouting presentation.",
  );
}

const careerLabels = [
  "Season",
  "Team(s)",
  "Pos",
  "Top archetype (season-specific)",
  "Confidence (%)",
  "Mixedness",
  "REG GP",
  "REG ATOI",
  "REG P",
  "REG G",
  "REG A",
  "REG SOG",
  "REG +/-",
  "REG PIM",
  "PO GP",
  "PO ATOI",
  "PO P",
  "PO G",
  "PO A",
  "PO SOG",
  "PO +/-",
  "PO PIM",
];
if (
  !appSource.includes(
    "How Does a Player's Play Style Evolve Over Their Career?",
  ) ||
  !indexSource.includes("<span>Career Trends</span>") ||
  !appSource.includes('career: "Career Trends"') ||
  indexSource.includes("<span>Career paths</span>") ||
  !appSource.includes(
    "Understanding the Evolution of a Player's Archetype",
  ) ||
  !appSource.includes("What Does the Evolution Really Mean?") ||
  !appSource.includes("What is Mixedness?") ||
  !appSource.includes("Stable top archetype + high confidence") ||
  !appSource.includes("Mixedness &gt;= <strong>0.40</strong>") ||
  !appSource.includes("<span class=\"field-label\">Group</span>") ||
  !appSource.includes("<h2 id=\"career-picker-title\">Select a player</h2>") ||
  !appSource.includes("<label for=\"career-search\">Search player name</label>") ||
  !appSource.includes("<label for=\"career-matches\">Matches</label>") ||
  !appSource.includes("Selected: ${escapeHTML(selected?.display || \"No player\")}") ||
  !appSource.includes("Seasons in dataset") ||
  !appSource.includes("Avg confidence") ||
  !appSource.includes("Avg mixedness") ||
  !appSource.includes("Hover over data points to see the full details.") ||
  !appSource.includes(
    "The circled points indicate years where there was a change in player archetype from the previous year.",
  ) ||
  !appSource.includes("Archetype and Career Stats by Season") ||
  careerLabels.some((label) => !appSource.includes(label)) ||
  !appSource.includes("function setupCareerTimeline(rows)") ||
  !appSource.includes("index > 0 && row.profile !== sorted[index - 1].profile") ||
  !appSource.includes("mean(rows.map((row) => row.confidencePct))") ||
  !appSource.includes("mean(rows.map((row) => row.mixedness))") ||
  !appSource.includes("data-career-point=") ||
  !appSource.includes("careerSeasonCard(row, index, rows.length)") ||
  appSource.includes("Style is a timeline, not a label.") ||
  !stylesSource.includes(".career-picker") ||
  !stylesSource.includes(".career-summary-grid") ||
  !stylesSource.includes(".career-chart-point.is-change::after") ||
  !stylesSource.includes(".career-season-card") ||
  !stylesSource.includes(".career-stat-grid")
) {
  throw new Error(
    "Career paths is missing Streamlit content, exact summary logic, or the accessible career-tape presentation.",
  );
}

const requiredCareerFields = [
  "season",
  "group",
  "id",
  "name",
  "team",
  "position",
  "profile",
  "confidence",
  "confidencePct",
  "mixedness",
  "games",
  "regAtoi",
  "points",
  "goals",
  "assists",
  "shots",
  "plusMinus",
  "pim",
  "playoffGames",
  "playoffAtoi",
  "playoffPoints",
  "playoffGoals",
  "playoffAssists",
  "playoffShots",
  "playoffPlusMinus",
  "playoffPim",
];
const careerKeys = new Set();
for (const row of careers) {
  const key = `${row.group}:${row.id}:${row.season}`;
  if (
    careerKeys.has(key) ||
    requiredCareerFields.some((field) => !(field in row)) ||
    !/^\d{2,}:\d{2}$/.test(row.regAtoi) ||
    !/^\d{2,}:\d{2}$/.test(row.playoffAtoi) ||
    !Number.isFinite(Number(row.confidencePct)) ||
    Number(row.confidencePct) < 0 ||
    Number(row.confidencePct) > 100 ||
    !Number.isFinite(Number(row.mixedness)) ||
    Number(row.mixedness) < 0 ||
    Number(row.mixedness) > 1
  ) {
    throw new Error(`${key} has incomplete or invalid career data.`);
  }
  careerKeys.add(key);
}
if (
  careers.length !== data.meta.playerSeasonCount ||
  careerKeys.size !== careers.length
) {
  throw new Error("Career history rows do not match the model-eligible seasons.");
}
const careerByKey = new Map(
  careers.map((row) => [
    `${row.group}:${row.id}:${row.season}`,
    row,
  ]),
);

const pierreLucDubois = careers
  .filter(
    (row) =>
      row.group === "forwards" &&
      Number(row.id) === 8479400,
  )
  .sort((a, b) => a.season.localeCompare(b.season));
const pierreLucConfidence =
  pierreLucDubois.reduce(
    (total, row) => total + Number(row.confidencePct),
    0,
  ) / pierreLucDubois.length;
const pierreLucMixedness =
  pierreLucDubois.reduce(
    (total, row) => total + Number(row.mixedness),
    0,
  ) / pierreLucDubois.length;
const pierreLuc2022 = pierreLucDubois.find(
  (row) => row.season === "20222023",
);
if (
  pierreLucDubois.length !== 9 ||
  Math.abs(pierreLucConfidence - 93.4) > 0.05 ||
  Math.abs(pierreLucMixedness - 0.066) > 0.0005 ||
  Number(pierreLuc2022?.confidencePct) !== 58.4 ||
  Number(pierreLuc2022?.mixedness) !== 0.416
) {
  throw new Error(
    "Career rounding no longer matches the Streamlit Pierre-Luc Dubois fixture.",
  );
}

const seasonPayloads = await Promise.all(
  data.meta.seasons.map(async ({ key }) => ({
    key,
    payload: JSON.parse(
      await readFile(`data/seasons/${key}.json`, "utf8"),
    ),
  })),
);
const seasonReadSignatures = new Set();
const seasonParagraphs = new Set();
for (const { key, payload } of seasonPayloads) {
  for (const group of ["forwards", "defense"]) {
    const read = payload[group]?.seasonRead;
    const groupProfileNames = payload[group].profiles.map(
      (profile) => profile.name,
    );
    const needFinder = payload[group]?.needFinder;
    const needTargets = needFinder?.targets || [];
    const targetProfiles = needTargets.map((target) => target.profile);
    const maxTargetCluster = Math.max(
      ...needTargets.map((target) => Number(target.cluster)),
    );
    if (
      !needTargets.length ||
      new Set(targetProfiles).size !== targetProfiles.length ||
      !needFinder?.details ||
      needTargets.some(
        (target) =>
          !target.profile ||
          !Number.isInteger(Number(target.cluster)) ||
          !needFinder.details[String(target.cluster)],
      )
    ) {
      throw new Error(
        `${key} ${group} is missing Need Finder target metadata.`,
      );
    }
    for (const player of payload[group].players) {
      const career = careerByKey.get(`${group}:${player.id}:${key}`);
      const careerPlayerFields = [
        ["name", "name"],
        ["team", "team"],
        ["position", "position"],
        ["profile", "profile"],
        ["confidence", "confidence"],
        ["games", "games"],
        ["regAtoi", "regAtoi"],
        ["points", "points"],
        ["goals", "goals"],
        ["assists", "assists"],
        ["shots", "shots"],
        ["plusMinus", "plusMinus"],
        ["pim", "pim"],
        ["playoffGames", "playoffGames"],
        ["playoffAtoi", "playoffAtoi"],
        ["playoffPoints", "playoffPoints"],
        ["playoffGoals", "playoffGoals"],
        ["playoffAssists", "playoffAssists"],
        ["playoffShots", "playoffShots"],
        ["playoffPlusMinus", "playoffPlusMinus"],
        ["playoffPim", "playoffPim"],
      ];
      const requiredPlayerFields = [
        "regAtoi",
        "playoffGames",
        "playoffAtoi",
        "playoffPoints",
        "playoffGoals",
        "playoffAssists",
        "playoffShots",
        "playoffPlusMinus",
        "playoffPim",
        "needOrder",
      ];
      if (
        !career ||
        careerPlayerFields.some(
          ([careerField, playerField]) =>
            career[careerField] !== player[playerField],
        ) ||
        !Array.isArray(player.targetScores) ||
        player.targetScores.length <= maxTargetCluster ||
        player.targetScores.some(
          (score) =>
            !Number.isFinite(Number(score)) ||
            Number(score) < 0 ||
            Number(score) > 100,
        ) ||
        requiredPlayerFields.some((field) => !(field in player)) ||
        !/^\d{2,}:\d{2}$/.test(player.regAtoi) ||
        !/^\d{2,}:\d{2}$/.test(player.playoffAtoi)
      ) {
        throw new Error(
          `${key} ${group} has incomplete Need Finder or career player data.`,
        );
      }
    }
    const constructions = payload[group]?.teamConstructions;
    const expectedRosterSize = group === "forwards" ? 12 : 8;
    const unitSize = group === "forwards" ? 3 : 2;
    if (!constructions || Object.keys(constructions).length < 20) {
      throw new Error(
        `${key} ${group} is missing team roster construction data.`,
      );
    }
    for (const [team, construction] of Object.entries(constructions)) {
      const units = construction.units || [];
      const rosterSize = units.reduce(
        (total, unit) => total + (unit.players?.length || 0),
        0,
      );
      if (
        units.length < 3 ||
        units.length > 4 ||
        rosterSize < unitSize * 3 ||
        rosterSize > expectedRosterSize ||
        units.some(
          (unit) =>
            !unit.players?.length ||
            unit.players.length > unitSize ||
            unit.players.some(
              (player) =>
                !player.id ||
                !player.name ||
                !player.profile,
            ),
        ) ||
        !construction.dominant?.profile ||
        !construction.mix?.length ||
        !construction.gaps?.length
      ) {
        throw new Error(
          `${key} ${group} ${team} has an incomplete roster construction.`,
        );
      }
      if (
        construction.usesMoneyPuck &&
        !units.some(
          (unit) =>
            unit.fromCombination &&
            Number.isFinite(Number(unit.xgPct)),
        )
      ) {
        throw new Error(
          `${key} ${group} ${team} is missing MoneyPuck unit metadata.`,
        );
      }
    }
    if (
      !read?.headline ||
      read.paragraphs?.length < 2 ||
      read.facts?.length < 3
    ) {
      throw new Error(`${key} ${group} is missing its season read.`);
    }
    seasonReadSignatures.add(
      `${read.headline}\n${read.paragraphs.join("\n")}`,
    );
    if (
      read.paragraphs.some(
        (paragraph) =>
          !groupProfileNames.some((name) => paragraph.includes(name)),
      )
    ) {
      throw new Error(
        `${key} ${group} has a paragraph without a current archetype mention.`,
      );
    }
    if (
      read.paragraphs.some((paragraph) =>
        /model learned|account for|combine for|top[- ]three|\d+\.\d+%/i.test(
          paragraph,
        ),
      )
    ) {
      throw new Error(
        `${key} ${group} fell back to data-regurgitation copy.`,
      );
    }
    read.paragraphs.forEach((paragraph) => seasonParagraphs.add(paragraph));
    if (
      read.comparison !== null ||
      read.metrics?.profileCount !== payload[group].profiles.length
    ) {
      throw new Error(
        `${key} ${group} contains an invalid cross-season interpretation.`,
      );
    }
  }
}
if (
  seasonReadSignatures.size !== data.meta.seasonCount * 2 ||
  seasonParagraphs.size !== data.meta.seasonCount * 4
) {
  throw new Error(
    "Every season read must contain two unique editorial paragraphs.",
  );
}

const latest = seasonPayloads.find(({ key }) => key === "20252026");
const ana = latest?.payload?.forwards?.teamConstructions?.ANA;
if (
  !ana ||
  ana.dominant.profile !== "High-Touch Risk/Reward Scorer" ||
  ana.dominant.overall !== 61.1 ||
  ana.dominant.top !== 67.5 ||
  ana.dominant.bottom !== 50.6 ||
  ana.units?.[0]?.players?.map((player) => player.name).join("|") !==
    "Chris Kreider|Leo Carlsson|Troy Terry" ||
  ana.units?.[0]?.minutes !== 469.5 ||
  ana.units?.[0]?.xgPct !== 0.591
) {
  throw new Error(
    "The Streamlit-parity ANA roster construction fixture changed unexpectedly.",
  );
}
const latestForwards = latest?.payload?.forwards;
const latestNeedTarget = latestForwards?.needFinder?.targets?.[0];
const latestNeedMatches = latestForwards?.players
  ?.filter((player) => Number(player.games) >= 20)
  .map((player) => ({
    ...player,
    similarity: Number(
      player.targetScores?.[Number(latestNeedTarget?.cluster)] || 0,
    ),
  }))
  .sort(
    (left, right) =>
      right.similarity - left.similarity ||
      Number(right.points || 0) - Number(left.points || 0) ||
      Number(left.needOrder || 0) - Number(right.needOrder || 0),
  )
  .slice(0, 80);
if (
  latestNeedTarget?.profile !== "Two-Way Shot-Share Driver" ||
  latestNeedTarget?.cluster !== 0 ||
  latestNeedMatches?.length !== 80 ||
  latestNeedMatches
    .slice(0, 5)
    .map((player) => player.name)
    .join("|") !==
    "Wyatt Johnston|Steven Stamkos|Adam Fantilli|Will Smith|Matty Beniers" ||
  latestNeedMatches.slice(0, 5).some(
    (player) => player.similarity !== 100,
  )
) {
  throw new Error(
    "The Streamlit-parity Need Finder fixture changed unexpectedly.",
  );
}

console.log(
  `Static site ready: ${data.meta.seasonCount} seasons, ` +
    `${data.meta.playerCount} players, ` +
    `${seasonReadSignatures.size} unique season reads.`,
);
